import agents.loader as loader


def test_discover_agents_non_empty():
    agents = loader.discover_agents()
    assert isinstance(agents, dict)
    # Expect at least agent_001 exists
    assert any(k.startswith('agent_001') or k == 'agent_001' for k in agents.keys())


def test_agent_config_contains_id():
    agents = loader.discover_agents()
    for k, v in agents.items():
        assert 'path' in v
        assert 'config' in v
        # config should be a dict
        assert isinstance(v['config'], dict)
        if v['config'].get('id'):
            assert isinstance(v['config']['id'], str)
