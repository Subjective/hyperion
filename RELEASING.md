# Releasing Hyperion-Opt to PyPI

This guide describes how to release new versions of `hyperion-opt` to PyPI.

## Prerequisites

### 1. GitHub Repository Setup
- Set up PyPI API tokens as GitHub secrets:
  - `PYPI_API_TOKEN` - for production PyPI
  - `TEST_PYPI_API_TOKEN` - for TestPyPI (optional)
  
To add these secrets:
1. Go to your repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add the token with the appropriate name

### 2. PyPI Account Setup
1. Create accounts on [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org)
2. Generate API tokens:
   - Go to Account Settings → API tokens
   - Create a token scoped to the `hyperion-opt` project (after first upload)

## Release Process

### 1. Prepare the Release

1. Update the version in `pyproject.toml`:
   ```toml
   version = "0.2.0"  # Update to your new version
   ```

2. Update the CHANGELOG (if you have one) with release notes

3. Run the full test suite:
   ```bash
   mise run check
   ```

4. Test the build locally:
   ```bash
   mise run build
   mise run check-package
   ```

5. Commit all changes:
   ```bash
   git add -A
   git commit -m "Prepare release v0.2.0"
   git push
   ```

### 2. Create a Release

#### Option A: Using Git Tags (Automated)
```bash
# Create an annotated tag
git tag -a v0.2.0 -m "Release version 0.2.0"

# Push the tag to GitHub
git push origin v0.2.0
```

This will trigger the GitHub Actions workflow to:
- Run the full test suite
- Build the distribution packages
- Upload to PyPI automatically

#### Option B: Using GitHub Releases UI
1. Go to your repository on GitHub
2. Click "Releases" → "Create a new release"
3. Choose a tag (create new: `v0.2.0`)
4. Set release title: "v0.2.0"
5. Add release notes
6. Click "Publish release"

### 3. Verify the Release

1. Check the GitHub Actions workflow:
   - Go to Actions tab
   - Verify the "Publish to PyPI" workflow succeeded

2. Verify on PyPI:
   ```bash
   # Check it's available
   pip search hyperion-opt  # Note: may be deprecated
   
   # Or try installing it
   pip install hyperion-opt==0.2.0
   ```

3. Test the installation:
   ```python
   from hyperion import tune
   print(hyperion.__version__)  # Should show 0.2.0
   ```

## Testing with TestPyPI

Before releasing to production, you can test with TestPyPI:

1. Build the package:
   ```bash
   mise run build
   ```

2. Upload to TestPyPI:
   ```bash
   mise run test-publish
   ```

3. Test installation from TestPyPI:
   ```bash
   pip install -i https://test.pypi.org/simple/ hyperion-opt
   ```

## Release Checklist

Use `mise run release-prep` to see an interactive checklist:

- [ ] Version bumped in `pyproject.toml`
- [ ] CHANGELOG updated
- [ ] All tests passing (`mise run check`)
- [ ] Package builds successfully (`mise run build`)
- [ ] Package validation passes (`mise run check-package`)
- [ ] Changes committed and pushed
- [ ] Tag created and pushed (or GitHub release created)
- [ ] GitHub Actions workflow completed successfully
- [ ] Package available on PyPI
- [ ] Installation tested from PyPI

## Versioning Guidelines

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

For pre-releases:
- Alpha: `0.2.0a1`
- Beta: `0.2.0b1`
- Release Candidate: `0.2.0rc1`

## Troubleshooting

### Build Failures
- Ensure all dependencies in `pyproject.toml` are correct
- Check that `MANIFEST.in` includes all necessary files
- Verify no syntax errors with `mise run check`

### Upload Failures
- Verify API tokens are correctly set in GitHub secrets
- Check that the package name isn't already taken
- Ensure version number is incremented from previous release

### Installation Issues
- Test with a clean virtual environment
- Verify all runtime dependencies are specified
- Check Python version compatibility

## Manual Publishing (Not Recommended)

If you need to publish manually for some reason:

```bash
# Build the distributions
python -m build

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# If everything looks good, upload to PyPI
twine upload dist/*
```

Note: Manual publishing requires PyPI credentials configured locally and is not recommended for production releases.