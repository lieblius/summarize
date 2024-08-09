run: setup
    @poetry run python -m summarize
setup:
    @command -v pipx &> /dev/null || (brew install pipx && pipx ensurepath && export PATH="$PATH:$(python3 -m site --user-base)/bin")
    @pipx list | grep poetry &> /dev/null || pipx install poetry
    @poetry check &> /dev/null || poetry install
update:
    @poetry install
fmt:
    @poetry run python -m black .
