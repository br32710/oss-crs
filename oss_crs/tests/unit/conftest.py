# SPDX-License-Identifier: MIT
"""Shared fixtures for unit tests."""

import pytest


@pytest.fixture(autouse=True)
def _cli_is_docker(request, monkeypatch):
    """Treat the container CLI as Docker without running `docker --version`.

    Tests that mock `subprocess.run` would otherwise see the probe. Tests of
    the probe itself opt out with `@pytest.mark.real_cli_probe`.
    """
    from oss_crs.src import docker_bin

    probe = docker_bin._cli_is_podman
    provider = docker_bin.compose_v2_provider
    probe.cache_clear()
    provider.cache_clear()
    if request.node.get_closest_marker("real_cli_probe") is None:
        monkeypatch.setattr(docker_bin, "_cli_is_podman", lambda: False)
    yield
    probe.cache_clear()
    provider.cache_clear()


@pytest.fixture(autouse=True)
def _multi_uid(request, monkeypatch):
    """Default to multi-UID mode so the host's subuid setup does not leak in.

    Tests of the detection itself opt out with `@pytest.mark.real_single_uid_probe`.
    """
    from oss_crs.src import docker_bin, ui
    from oss_crs.src.templates import renderer

    detect = docker_bin.single_uid_mode_enabled
    detect.cache_clear()
    if request.node.get_closest_marker("real_single_uid_probe") is None:
        for module in (docker_bin, ui, renderer):
            monkeypatch.setattr(module, "single_uid_mode_enabled", lambda: False)
    yield
    detect.cache_clear()
