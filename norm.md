# 42 Coding Norm (Norminette) Rules

This document outlines the coding standards and formatting constraints that must be followed for C programming projects.

---

## Project Deviations

### Compliance Scope

The active norm-compliance scope is limited to production solver code:

* `src/`
* `include/`
* `Makefile`
* `tools/makefile/`

The following directories are explicitly out of scope because they contain
historical experiments, benchmark prototypes, generated data, or external
workflow helpers:

* `experiments/`
* `exp_dist_calc/`
* `instances/`
* `results/`
* `docs/`
* `tools/python/`
* `tools/bash/`
* `tools/batch/`

This repository does not enforce the traditional 42 preference for replacing
`for` loops with `while` loops. `for` loops are allowed when they improve
readability, especially for simple counted iteration.

Allowed `for` loops must still respect the rest of this document:

* Do not declare loop variables inside the `for` header.
* Do not combine declaration and initialization.
* Keep declarations at the beginning of the function or scope.
* Keep one instruction per line and avoid assignments in conditions.

Example:

```c
int	i;

i = 0;
for (; i < n; i++)
{
	/* ... */
}
```

---

## III.1 Naming Conventions

* **Structures**: A structure’s name must start with `s_` (e.g., `struct s_var`).
* **Typedefs**: A typedef’s name must start with `t_` (e.g., `t_var`).
* **Unions**: A union’s name must start with `u_` (e.g., `union u_var`).
* **Enums**: An enum’s name must start with `e_` (e.g., `enum e_var`).
* **Globals**: A global’s name must start with `g_` (e.g., `g_var`).
* **Identifiers**: Variable names, function names, and user-defined types can only contain lowercase letters, digits, and underscores (`_`) (**snake_case**). No capital letters are allowed.
* **Files & Directories**: Names can only contain lowercase letters, digits, and underscores (`_`) (**snake_case**).
* **Character Set**: Characters that aren’t part of the standard ASCII table are forbidden, except inside literal strings and character constants.
* **Language & Clarity**: All identifiers (functions, types, variables, macros, filenames, and directories) must be explicit, mnemonic, and readable in English, with words separated by underscores.
* **Global Variables**: Using global variables that are not marked `const` or `static` is forbidden and is considered a norm error, unless the project explicitly allows them.
* **Compilation**: The code must compile. A file that doesn't compile cannot pass the Norm.

---

## III.2 Formatting Rules

* **Function Length**: Each function must be at most **25 lines** long, not counting the function’s own opening and closing braces.
* **Line Width**: Each line must be at most **80 columns** wide, comments included.
  > [!IMPORTANT]
  > A tabulation character (`\t`) counts as the number of spaces it represents in the editor (usually 4 or 8), not as a single column.
* **Function Separation**: Functions must be separated by at least one empty line. Comments or preprocessor instructions can be inserted between functions.
* **Indentation**: You must indent your code with **4-char-long tabulations** (ASCII character code 9). Real tabulations must be used, not spaces.
* **Braces**: Blocks within braces must be indented. Braces must be alone on their own line, except in declarations of `struct`, `enum`, or `union`.
* **Empty Lines**: An empty line must be completely empty: no trailing spaces or tabulations.
* **Trailing Whitespace**: A line can never end with spaces or tabulations.
* **Spacing Rules**:
  * You can never have two consecutive empty lines.
  * You can never have two consecutive spaces.
* **Variable Declarations**:
  * All declarations must be at the beginning of a function/scope.
  * All variable names in the same scope must be aligned/indented on the same column.
  * The asterisk (`*`) for pointers must be stuck to the variable name (e.g., `int *ptr`, not `int* ptr` or `int * ptr`).
  * Only one variable declaration is allowed per line.
  * Declaration and initialization cannot be on the same line, except for global variables (when allowed), static variables, and constants.
* **Internal Function Spacing**: In a function, you must place exactly **one empty line** between variable declarations and the remaining code. No other empty lines are allowed inside a function body.
* **Instructions Per Line**: Only one instruction or control structure per line is allowed.
  * Assignment inside a control structure condition is forbidden.
  * Multiple assignments on the same line are forbidden.
  * A newline is required at the end of a control structure.
* **Line Splitting**: Instructions can be split into multiple lines when needed. Subsequent lines must be indented relative to the first line. Use operators at the beginning of the new line (not at the end of the previous one).
* **Punctuation Spacing**: Unless it is the end of a line, each comma or semicolon must be followed by a space.
* **Operator Spacing**: Each operator or operand must be separated by exactly one space.
* **Keyword Spacing**: Each C keyword must be followed by a space, except for keywords for types (such as `int`, `char`, `float`, etc.) and `sizeof`.
* **Brace Requirements**: Control structures (`if`, `while`, etc.) must use braces, unless they contain a single instruction on a single line.

---

## III.3 Function Restrictions

* **Parameter Limit**: A function can take at most **4 named parameters**.
* **Void Arguments**: A function that doesn't take arguments must be explicitly prototyped with `void` (e.g., `int func(void)`).
* **Prototype Names**: Parameters in function prototypes must be named.
* **Variable Limit**: You cannot declare more than **5 variables** per function.
* **Return Syntax**: The return value of a function must be enclosed in parentheses, unless the function returns nothing (e.g., `return (value);`).
* **Name Tabulation**: Each function definition/prototype must have exactly **one tabulation** between its return type and its name (e.g., `int\tfunc(void)`).

---

## III.4 Typedefs, Structs, Enums, and Unions

* **Keyword Spacing**: Add a space between `struct`/`enum`/`union` and its name when declaring it.
* **Variable Indentation**: Apply the usual variable indentation rules when declaring structure, enum, or union variables.
* **Internal Struct Spacing**: Regular indentation rules apply inside the braces of a `struct`, `enum`, or `union`.
* **Typedef Spacing**: Add a space after `typedef`, and apply regular indentation for the newly defined name.
* **Struct Alignment**: You must indent all structure members' names on the same column in their scope.
* **Scope Restriction**: You cannot declare a structure inside a `.c` file; they must be declared in header files.

---

## III.5 Header Files (`.h`)

* **Allowed Elements**: Header inclusions (system or user), declarations, defines, prototypes, and macros.
* **Inclusion Position**: All `#include` directives must be at the very beginning of the file.
* **No C File Inclusions**: You cannot include a `.c` file in a header file or in another `.c` file.
* **Double Inclusion Protection**: Headers must be protected with include guards. For a file named `ft_foo.h`, the guard macro must be `FT_FOO_H`.
* **Unused Inclusions**: Inclusion of unused headers is strictly forbidden.
* **Inclusion Comments**: Header inclusions can be justified in `.c` and `.h` files using comments.

---

## III.7 Macros and Pre-processors

* **Pre-processor Constants**: `#define` constants must only be used for literal and constant values.
* **No Norm Bypassing**: Defining macros to bypass the norm or obfuscate code is strictly forbidden.
* **Standard Macros**: You can use macros from standard libraries only if they are allowed in the project scope.
* **Formatting Limits**:
  * Multiline macros are forbidden.
  * Macro names must be entirely uppercase (`UPPERCASE`).
* **Guard Indentation**: Preprocessor directives must be indented inside `#if`, `#ifdef`, or `#ifndef` blocks.
* **Global Scope Only**: Preprocessor instructions are forbidden outside of the global scope.

---

## III.9 Comments

* **Positioning**: Comments cannot be placed inside function bodies. They must either be at the end of a line or on their own line outside functions.
* **Language & Quality**: Comments must be in English and must be useful.
* **No Bad Code Justification**: Comments cannot be used to justify bad design or poorly written functions.

---

## III.10 Files & Makefile Rules

* **C File Inclusions**: You cannot include a `.c` file in another `.c` file.
* **Function Limit**: You cannot have more than **5 function definitions** in a single `.c` file.
* **Makefile Requirements**:
  * Mandatory rules: `$(NAME)`, `clean`, `fclean`, `re`, and `all`.
  * The `all` rule must be the default rule (executed when running `make` with no arguments).
  * The Makefile must not relink targets when no source files have changed.
  * For multi-binary projects, a specific rule for each binary must exist (e.g., `$(NAME_1)`, `$(NAME_2)`), and `all` must compile them using their respective rules.
  * If the project depends on a non-system library (e.g., `libft`) present in the source directory, the Makefile must compile it automatically.
  * All source files needed to compile the project must be explicitly listed in the Makefile (wildcards like `*.c` or `*.o` are forbidden).
