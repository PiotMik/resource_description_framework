import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


class CompositeCalculation(nx.DiGraph):

    def __init__(self, name, edges, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.add_edges_from(edges)

    def vizualize(self, *args, **kwargs):
        default_kwargs = {'arrows': True, 'pos': nx.fruchterman_reingold_layout(self),
                          'node_color': 'white'}
        default_kwargs.update(**kwargs)
        nx.draw_networkx(self, *args, **default_kwargs)
        plt.title(f"Schema of {self.name}")

    def get_sorted_node_predecessors(self, node):
        if node.check_input_order:
            inputs = {input_node: args['input_idx'] for input_node, args in self.pred[node].items()}
            inputs = sorted(inputs.items(), key=lambda node_info: node_info[1])
            inputs = [input_node for input_node, _ in inputs]
        else:
            inputs = list(self.pred[node])
        return inputs

    def get_sorted_node_successors(self, node):
        if node.check_output_order:
            successors = {output_node: args['output_idx'] for output_node, args in self.succ[node].items()}
            successors = sorted(successors.items(), key=lambda node_info: node_info[1])
            successors = [successor_node for successor_node, _ in successors]
        else:
            successors = list(self.succ[node])
        return successors

    def _get_predecessor_edge_information(self, node, key):
        predecessor_nodes = self.get_sorted_node_predecessors(node)
        inputs = [self.edges[predecessor, node][key] for predecessor in predecessor_nodes]
        return inputs

    def _update_successor_edge_information(self, node, key, values):
        successors = self.get_sorted_node_successors(node)
        for successor_node, value in zip(successors, values):
            self.edges[node, successor_node][key] = value

    def _unvisited_nodes_with_all_inputs(self, edge_attribute):
        nodes_with_all_inputs = set()
        for node in self.nodes:
            is_final_node = node == self.output_node
            node_visited = all([self.edges[node, succ][edge_attribute] is not None for succ in self.successors(node)])
            node_inputs_ready = all([self.edges[pred,
                                                node][edge_attribute] is not None for pred in self.predecessors(node)])
            if node_inputs_ready and (not node_visited or is_final_node):
                nodes_with_all_inputs |= {node}
        return list(nodes_with_all_inputs)

    def _clear_cache(self):
        for node in self.nodes:
            if "_result_cache" in node.__dict__:
                node._result_cache = None

    def _clear_edge_info(self, key):
        clean_state = {edge: {key: None} for edge in self.edges}
        nx.set_edge_attributes(self, clean_state)

    @property
    def input_nodes(self):
        return [node for node, predecessors in self.pred.items() if len(predecessors) == 0]

    @property
    def output_node(self):
        nodes_without_successors = [node for node, successors in self.succ.items() if len(successors) == 0]
        if len(nodes_without_successors) != 1:
            raise ValueError(f"The graph needs to have exactly one output node. Got: {nodes_without_successors}")
        return nodes_without_successors[0]

    def compute(self, input_data: pd.DataFrame):
        self._clear_cache()
        self._clear_edge_info('compute')
        for node in self.input_nodes:
            calculation_results = node.compute(input_data)
            self._update_successor_edge_information(node=node,
                                                    key='compute',
                                                    values=[calculation_results])
        return self._traverse_graph(method_name="compute")

    def parse_frequencies(self, input_frequencies: dict):
        self._clear_edge_info('output_frequency')
        for node in self.input_nodes:
            calculation_results = node.output_frequency(input_frequencies[node.ts_name])
            self._update_successor_edge_information(node=node,
                                                    key='output_frequency',
                                                    values=[calculation_results])
        return self._traverse_graph(method_name="output_frequency")

    def _traverse_graph(self, method_name, max_iter=10):
        final_node = self.output_node
        computable_nodes = self._unvisited_nodes_with_all_inputs(method_name)
        final_node_reached = False
        iters_left = max_iter
        while len(computable_nodes) > 0 and iters_left > 0:
            for node in computable_nodes:
                inputs = self._get_predecessor_edge_information(node, method_name)
                try:
                    calculation_results = getattr(node, method_name)(*inputs)
                except ValueError as err:
                    raise ValueError(f"Problem occured when evaluating node {node}: {err}")
                if not isinstance(calculation_results, list):
                    calculation_results = [calculation_results]

                if node != final_node:
                    self._update_successor_edge_information(node=node,
                                                            key=method_name,
                                                            values=calculation_results)
                else:
                    final_node_reached = True
                    break
            computable_nodes = self._unvisited_nodes_with_all_inputs(method_name)
            iters_left -= 1
        if not final_node_reached:
            raise ValueError(f"Couldn't reach the final node.")

        return calculation_results


if __name__ == "__main__":
    import numpy as np
    from nodes import Subtract, Output, Input, Add, Aggregate

    # Some input data
    n_ts, n_dt = (4, 10)
    sample_data = pd.DataFrame(data=np.linspace(1, n_dt, n_dt).reshape(-1, 1).repeat(n_ts, 1).cumsum(axis=1),
                               columns=[f"TS{i + 1}" for i in range(n_ts)].copy(),
                               index=pd.period_range(start=pd.Period('2020-01', 'M'),
                                                     end=pd.Period('2020-01', 'M') + n_dt - 1,
                                                     freq="M"))

    # Definition of computation units
    sub1 = Subtract(const_float=0.0)
    out = Output('NEW')
    add1 = Add(const_float=5.0)
    aggr1 = Aggregate(frequency='Q', aggregation="max")

    # Definition of the calculation graph
    e = [(Input("TS2"), sub1, {'input_idx': 2}),
         (Input("TS1"), sub1, {'input_idx': 1}),
         (sub1, add1),
         (Input("TS3"), add1),
         (add1, aggr1),
         (aggr1, out)]
    G = CompositeCalculation(name="MyFirstCalc", edges=e)

    # User can validate the flow of frequencies
    G.parse_frequencies(input_frequencies={'TS2': 'Q',
                                           'TS1': 'M',
                                           'TS3': 'Q'})

    # User can inspect the graph
    G.vizualize()

    # User can run the computation
    G.compute(sample_data)
