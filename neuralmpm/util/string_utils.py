def ids_to_list(id_list_str: str) -> list:
    """
    Convert a string of IDs to a list of integers.

    e.g.:
        '1,2,3'    -> [1, 2, 3]
        '1-3'      -> [1, 2, 3]
        '1,2,8-10' -> [1, 2, 8, 9, 10]

    Args:
        id_list_str: Formatted string of IDs.

    Returns:
        The list of IDs. (list of int)

    """
    id_list_str = id_list_str.split(",")
    id_list = []

    for id_ in id_list_str:
        if "-" in id_:
            start, end = id_.split("-")
            id_list += list(range(int(start), int(end) + 1))
        else:
            id_list.append(int(id_))

    return id_list
