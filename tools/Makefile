SHELL := /bin/bash

.PHONY: context7-setup
context7-setup:
	chmod +x bootstrap_context7_key.sh
	./bootstrap_context7_key.sh --out .mcp_env

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  context7-setup   Make the bootstrap script executable and run it (writes tools/.mcp_env)"
