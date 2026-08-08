def validate_structural_path(
    graph,
    path,
    expected_origin,
    expected_destination,
) -> tuple[bool, str]:
    if path is None:
        return False, "empty_path"
    if not isinstance(path, (list, tuple, str)): # allow strings if it's somehow stringified? No, strictly lists.
        # But wait, pymoo X might contain objects. 
        if isinstance(path, str):
            try:
                import ast
                path = ast.literal_eval(path)
            except:
                pass
    if not isinstance(path, (list, tuple)):
        return False, "malformed_path"
    if len(path) < 2:
        return False, "empty_path"
    if str(path[0]) != str(expected_origin):
        return False, "wrong_origin"
    if str(path[-1]) != str(expected_destination):
        return False, "wrong_destination"
        
    for node in path:
        if node not in graph:
            return False, "unknown_node"
            
    for u, v in zip(path[:-1], path[1:]):
        if not graph.has_edge(u, v):
            return False, "missing_edge"
            
    if len(set(path)) != len(path):
        return False, "has_cycles"
            
    return True, "valid"
