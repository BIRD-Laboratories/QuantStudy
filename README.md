### Usage Guide

#### Installation

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/BIRD-Laboratories/QuantStudy/
   cd QuantStudy
   ```

#### Running the Simulation

1. **Navigate to the Project Directory:**
   ```sh
   cd QuantStudy
   ```

2. **Run the Main Script:**
   ```sh
   python main.py --params_dir parameters.json
   ```

   - `--params_dir`: Path to the JSON file containing simulation parameters. Default is `parameters.json`.

#### Output

- The simulation will generate a plot of the economic indicators over time and save it as `economics.png`.

### Contributing Guide

#### Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

#### How to Contribute

1. **Fork the Repository:**
   - Click the "Fork" button on the top right of the repository page.

2. **Clone Your Fork:**
   ```sh
   git clone https://github.com/yourusername/QuantStudy.git
   cd QuantStudy
   ```

3. **Create a New Branch:**
   ```sh
   git checkout -b your-branch-name
   ```

4. **Make Your Changes:**
   - Follow PEP8.
   - Ensure your changes are well-documented.

5. **Run Tests:**
   ```sh
   python -m unittest discover tests
   ```

6. **Commit Your Changes:**
   ```sh
   git commit -m "Your detailed description of your changes."
   ```

7. **Push to Your Fork:**
   ```sh
   git push origin your-branch-name
   ```

8. **Submit a Pull Request:**
   - Go to the original repository and click the "New Pull Request" button.
   - Describe your changes in detail and submit the pull request.

#### Reporting Issues

- If you find a bug or have a feature request, please open an issue on the [GitHub Issues page](https://github.com/BIRD-Laboratories/QuantStudy/issues).
- Provide detailed information about the issue, including steps to reproduce it if applicable.

#### Code Style

- Follow the PEP 8 style guide for Python code.
- Use meaningful variable and function names.
- Document your code using docstrings and comments where necessary.

#### Testing

- Write unit tests for new functionality.
- Ensure all existing tests pass before submitting a pull request.

#### Review Process

- All contributions will be reviewed by the maintainers.
- Once approved, your contribution will be merged into the main branch.

Thank you for your interest in contributing to the `QuantStudy` project!
