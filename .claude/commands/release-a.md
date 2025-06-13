# Alpha Release Command

Run the complete alpha release process automatically:

1. Run fast tests to ensure code quality
2. Run linting and type checking 
3. Determine the next alpha version number automatically
4. Create and push the git tag to trigger the release

Please execute the following steps in order:

1. First, run the fast test suite: `uv run pytest -m "not slow" -k "not integration"`
2. Run linting with auto-fix: `uv run ruff check --fix`  
3. Run type checking: `uv run mypy`
4. Check the latest alpha tag: `git tag --sort=-version:refname | grep "^v.*a[0-9]*$" | head -1`
5. Determine the next alpha version by incrementing the alpha number
6. Create the new tag: `git tag v0.1.0a[N]` (where N is the next number)
7. Push the tag to trigger release: `git push origin v0.1.0a[N]`

After completion, confirm that all tests passed, the tag was created successfully, and the GitHub release workflow has started.