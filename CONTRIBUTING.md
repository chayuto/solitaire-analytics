# Contributing to Solitaire Analytics Engine

We welcome contributions to the Solitaire Analytics Engine! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/solitaire-analytics.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `pip install -r requirements.txt`
5. Install the package in development mode: `pip install -e .`

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m models
pytest -m engine
pytest -m solver
pytest -m analysis

# Run with coverage
pytest --cov=solitaire_analytics
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Keep functions focused and modular

### Testing

- Write tests for all new features
- Ensure existing tests pass
- Aim for >70% code coverage
- Use appropriate pytest markers (unit, integration, slow, gpu, etc.)

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update tests to cover your changes
3. Ensure all tests pass
4. Update documentation as needed
5. Create a pull request with a clear description of your changes

## Test Markers

Use these pytest markers to categorize your tests:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.solver` - Solver-specific tests
- `@pytest.mark.analysis` - Analysis component tests
- `@pytest.mark.models` - Model tests
- `@pytest.mark.engine` - Engine tests

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Keep discussions professional

## Questions?

If you have questions, please open an issue with the "question" label.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
