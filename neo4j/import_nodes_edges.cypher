"""USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///graph/nodes.csv' AS row
MERGE (n {id: row.id})
SET n :`Node`, n += {
  node_type: row.node_type,
  name: row.name,
  props: apoc.convert.fromJsonMap(row.props)
};

USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///graph/edges.csv' AS row
MATCH (a {id: row.source}), (b {id: row.target})
CALL apoc.create.relationship(a, row.rel_type, apoc.convert.fromJsonMap(row.props), b) YIELD rel
RETURN count(*);
"""