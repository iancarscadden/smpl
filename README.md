
# SMPL Language Interpreter

![SMPL Logo](logo.png)

**SMPL** is a simple, educational programming language designed for learning how interpreters and languages are built. It provides a minimal yet flexible language structure with classes, functions, variables, loops, conditionals, and more. This project includes the interpreter for SMPL, along with a built-in library for list operations (`smpl_lists.py`). Users can write `.smpl` scripts and run them through this interpreter.

## Features

- **Classes**: Define classes with properties and methods.
- **Functions**: Create your own functions with parameters and return values.
- **Control Flow**: Use `if`, `elif`, `else`, `for`, and `while` for complex logic.
- **Built-in Functions**: Includes `print`, math functions (`sqrt`, `pow`, etc.), and `get_input`.
- **Lists**: A built-in `List` class for handling collections of items.
- **Error Handling**: Clear error messages for syntax and runtime errors.

## Getting Started

### Prerequisites

- **Python 3.10+**: Ensure you have a compatible version of Python installed.
- **Pip**: The standard Python package manager.

### Installation

You can install SMPL via `pip`. From the project directory (where `setup.py` is located), run:

```bash
pip install .
```

This command will build and install the `smpl` command-line tool on your system. After installation, you can run SMPL files using:

```bash
smpl <filename.smpl>
```

### Verify Installation

To verify that the `smpl` command is available, run:

```bash
smpl --help
```

If installed correctly, you'll see usage information and a brief help message.

## Usage

1. **Write your SMPL program** in a file with a `.smpl` extension. For example:

   ```smpl
   class Person {
       name = ""
       age = ""
       height = ""

       func greet {
           print("Hello, my name is " + name)
       }
   }

   bob = new Person
   bob.name = "Bob"
   bob.greet()
   ```

2. **Run the program**:

   ```bash
   smpl example.smpl
   ```

   You should see:

   ```
   Hello, my name is Bob
   ```

### Other Examples

- **Variables & Math**:
   ```smpl
   x = 5
   y = 2
   z = x * y + 10
   print(z)
   ```

- **Lists**:
   ```smpl
   list nums = (1, 2, 3)
   nums.append(4)
   print(nums.get(0))
   print(nums.size())
   ```

- **Functions & Control Flow**:
   ```smpl
   func add using a, b {
       return a + b
   }

   result = add(10, 20)
   print(result)  // Prints 30

   if result > 25 {
       print("Result is greater than 25")
   } else {
       print("Result is not greater than 25")
   }
   ```

## Project Structure

- **interpreter.py**: The main interpreter code. Handles tokenization, parsing, evaluation, and execution.
- **smpl_lists.py**: Provides a `List` class and related methods for use within SMPL.
- **setup.py**: Packaging and installation configuration.

## Development

1. **Cloning the Repository**:

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

4. **Contributions**:  
   If you'd like to contribute:
   - Fork the repository
   - Create a feature branch
   - Make changes and add tests
   - Submit a pull request

## Known Issues & Limitations

- Limited standard library beyond basics and lists.
- Error messages may still be improved for certain edge cases.

## License

This project is released under the [MIT License](LICENSE). Feel free to use and modify it for educational or personal projects.

---

**Enjoy experimenting with SMPL!** If you run into issues or have suggestions, feel free to open an issue on the repository.
