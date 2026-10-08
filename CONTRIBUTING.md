# Contributing to Model Plane

Thank you for your interest in contributing to Model Plane! We welcome contributions from the community.

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/model-plane.git
   cd model-plane
   ```
3. **Set up Python environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev,all-classifiers]"
   ```
4. **Set up UI dependencies**:
   ```bash
   pnpm install
   ```

## Development Workflow

- Start the full stack (API + UI dev server with hot reload):
  ```bash
  pnpm start
  ```
- Run tests:
  ```bash
  pnpm test
  # or fast test without coverage
  pnpm run test:fast
  ```
- Linting & formatting:
  ```bash
  ruff check .
  ruff format --check .
  ```

## Pull Request Guidelines

1. Create a branch for your feature or bugfix (`git checkout -b feature/my-feature`).
2. Ensure all tests pass (`pnpm test`).
3. Write clean, readable code and include unit tests for new behavior or fixes.
4. Keep PRs focused on a single topic.
5. Submit a Pull Request with a clear description of the problem and your solution.

## Code of Conduct

Please be respectful and constructive in all discussions, pull requests, and issues.
