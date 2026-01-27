import json
import csv
import pytest

@pytest.mark.skipif(not pytest.importorskip('jsonschema'), reason="jsonschema not available")
def test_schema_and_files():
    # Load schemas
    with open('schemas/tranche.json') as f:
        tranche_schema = json.load(f)

    with open('schemas/tranche_run.json') as f:
        tranche_run_schema = json.load(f)

    # Validate schemas
    example_object = {"id": 1, "node_type": "example", "name": "example node", "props": {}}
    jsonschema.validate(instance=example_object, schema=tranche_schema)
    jsonschema.validate(instance=example_object, schema=tranche_run_schema)

    # Check that files exist and have correct headers
    for filename, expected_headers in {
        'graph/nodes.csv': ['id', 'node_type', 'name', 'props'],
        'graph/edges.csv': ['source', 'target', 'rel_type', 'props']
    }.items():
        with open(filename, newline='') as f:
            reader = csv.DictReader(f)
            actual_headers = reader.fieldnames
            assert actual_headers == expected_headers, f"{filename} headers do not match!"

    # Check Neo4j files
    for filename, expected_keyword in {
        'neo4j/constraints.cypher': 'CREATE CONSTRAINT',
        'neo4j/import_nodes_edges.cypher': 'LOAD CSV'
    }.items():
        with open(filename) as f:
            content = f.read()
            assert expected_keyword in content,

        f"{filename} does not contain {expected_keyword}!"