import rdflib

file_path = "dataset/sample.ttl"
g = rdflib.Graph()
g.parse(file_path, format="turtle")

results = g.query("""
SELECT ?x
WHERE {
?x rdfs:subClassOf <#mnemonic>.
}
""")

for r in results:
    print(r)