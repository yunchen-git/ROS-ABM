import species_registry as registry


def test_species_registry_preserves_canonical_names_and_indices():
    assert registry.INTERC_SPECIES_CAMP == "cAMP"
    assert registry.INTERC_SPECIES_H2O2I == "H2O2i"
    assert registry.INTERC_SPECIES_CA == "Ca"
    assert registry.INTERC_SPECIES_ORDER == ("cAMP", "H2O2i", "Ca")
    assert registry.EXT_GRADIENT_H2O2E == "H2O2e"

    assert registry.ODE_STATE_H2O2E == 0
    assert registry.ODE_STATE_H2O2I == 1
    assert registry.ODE_STATE_CA_I == 29
    assert registry.ODE_STATE_CA_E == 30
    assert registry.ODE_STATE_CA_ER == 31


def test_species_registry_role_mappings_preserve_current_roles():
    assert registry.INTERC_SPECIES_BY_ROLE == {
        "camp": "cAMP",
        "h2o2_i": "H2O2i",
        "ca_i": "Ca",
    }
    assert registry.ODE_STATE_INDEX_BY_ROLE == {
        "h2o2_e": 0,
        "h2o2_i": 1,
        "ca_i": 29,
        "ca_e": 30,
        "ca_er": 31,
    }
