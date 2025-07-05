# Contributing to CMG-X

Thank you for your interest in contributing to CMG-X! This document provides guidelines for different types of contributions.

## 🔬 Research Contributions

### Adding New Pooling Methods
1. Implement the method in `cmgx/pooling/`
2. Add PyG integration wrapper
3. Include in benchmark scripts
4. Document performance characteristics

### Experimental Extensions
1. Create new benchmark script in `examples/`
2. Follow existing naming convention
3. Include statistical validation
4. Document methodology in README

## 💻 Software Contributions

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints where appropriate
- Include docstrings for all public functions
- Write unit tests for new features

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_new_feature.py
```

## 📊 Benchmarking Guidelines

### Adding New Datasets
1. Update benchmark scripts to include new dataset
2. Verify fair comparison protocols
3. Document dataset characteristics
4. Include statistical significance tests

### Performance Evaluation
- Use multiple random seeds (minimum 3)
- Include proper validation methodology
- Report statistical significance
- Document computational requirements

## 📝 Documentation

### README Updates
- Keep experimental results current
- Document new features clearly
- Include usage examples
- Update citation information

### Code Documentation
- Include clear docstrings
- Add inline comments for complex logic
- Provide usage examples
- Document performance characteristics

## 🐛 Bug Reports

Use the following template for bug reports:

```
**Bug Description**
Clear description of the bug

**Reproduction Steps**
1. Step 1
2. Step 2
3. ...

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: 
- Python version:
- PyTorch version:
- PyG version:
- CMG-X version:

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

Use the following template for feature requests:

```
**Feature Description**
Clear description of the proposed feature

**Motivation**
Why is this feature needed?

**Proposed Implementation**
How should this be implemented?

**Alternatives Considered**
What other approaches were considered?

**Additional Context**
Any other relevant information
```

## 🔄 Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation as needed
7. Commit changes (`git commit -m 'Add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Pull Request Guidelines
- Provide clear description of changes
- Reference any related issues
- Include test results
- Update documentation
- Ensure code follows style guidelines

## 📧 Questions?

For questions about contributing:
- Open a GitHub Discussion
- Contact maintainers directly
- Join our research community

Thank you for helping make CMG-X better! 🚀
