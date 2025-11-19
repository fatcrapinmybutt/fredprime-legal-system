from agents import menu


def test_menu_items_structure():
    items = menu.get_menu_items()
    assert isinstance(items, list)
    if items:
        it = items[0]
        assert 'id' in it and 'name' in it and 'description' in it and 'path' in it
