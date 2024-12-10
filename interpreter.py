import sys
import re
import math
import smpl_lists  # Import the smpl_lists module

def get_input_func():
    try:
        user_input = input()
        # Try to convert to int or float if possible
        if user_input.isdigit():
            return int(user_input)
        try:
            return float(user_input)
        except ValueError:
            return user_input  # Return as a string if it can't be converted
    except EOFError:
        return ""

FUNCTIONS = {
    'sqrt': math.sqrt,
    'pow': math.pow,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'get_input': get_input_func,
}

FUNCTIONS_USER = {}
CLASSES = {}

RESERVED_KEYWORDS = {'print', 'if', 'elif', 'else', 'for', 'while', 'func', 'return', 'class', 'new', 'list'}

def tokenize(expression):
    token_specification = [
        ('NUMBER',   r'\d+(\.\d+)?'),
        ('BOOLEAN',  r'true|false'),
        ('OPERATOR', r'==|!=|<=|>=|and|or|<|>|\+|\-|\*|\/|\%|\^'),
        ('DOT',      r'\.'),
        ('ELIF',     r'elif'),
        ('ELSE',     r'else'),
        ('IF',       r'if'),
        ('CLASS',    r'class'),
        ('NEW',      r'new'),
        ('LIST',     r'list'),
        ('STRING',   r'"[^"]*"'),
        ('IDENT',    r'[A-Za-z_]\w*'),
        ('LPAREN',   r'\('),
        ('RPAREN',   r'\)'),
        ('COMMA',    r','),
        ('SKIP',     r'[ \t]+'),
        ('MISMATCH', r'.'),
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    get_token = re.compile(tok_regex).match
    pos = 0
    tokens = []
    mo = get_token(expression, pos)
    while mo is not None:
        kind = mo.lastgroup
        value = mo.group(kind)
        if kind == 'NUMBER':
            tokens.append({'type': 'NUMBER', 'value': float(value) if '.' in value else int(value)})
        elif kind == 'BOOLEAN':
            tokens.append({'type': 'BOOLEAN', 'value': True if value == 'true' else False})
        elif kind == 'STRING':
            tokens.append({'type': 'STRING', 'value': value[1:-1]})
        elif kind == 'IDENT':
            if value in RESERVED_KEYWORDS:
                tokens.append({'type': value.upper(), 'value': value})
            else:
                tokens.append({'type': 'IDENT', 'value': value})
        elif kind in {'IF', 'ELIF', 'ELSE', 'CLASS', 'NEW', 'LIST'}:
            tokens.append({'type': kind, 'value': value})
        elif kind == 'DOT':
            tokens.append({'type': 'DOT', 'value': value})
        elif kind == 'OPERATOR':
            tokens.append({'type': 'OPERATOR', 'value': value})
        elif kind == 'LPAREN':
            tokens.append({'type': 'LPAREN', 'value': value})
        elif kind == 'RPAREN':
            tokens.append({'type': 'RPAREN', 'value': value})
        elif kind == 'COMMA':
            tokens.append({'type': 'COMMA', 'value': value})
        elif kind == 'SKIP':
            pass
        elif kind == 'MISMATCH':
            raise ValueError(f'Unexpected character "{value}" in expression.')
        pos = mo.end()
        mo = get_token(expression, pos)
    return tokens

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, ttype=None):
        tok = self.current_token()
        if tok is None:
            raise ValueError("Unexpected end of input")
        if ttype and tok['type'] != ttype:
            raise ValueError(f"Expected {ttype}, got {tok['type']}")
        self.pos += 1
        return tok

    def match(self, ttype, value=None):
        tok = self.current_token()
        if tok and tok['type'] == ttype:
            if value is not None:
                return tok['value'] == value
            return True
        return False

    def match_op(self, ops):
        tok = self.current_token()
        if tok and tok['type'] == 'OPERATOR':
            return tok['value'] in ops
        return False

    def parse_expression(self):
        return self.parse_logical_expr()

    def parse_logical_expr(self):
        node = self.parse_equality_expr()
        while self.match_op(['and','or']):
            op = self.consume('OPERATOR')
            right = self.parse_equality_expr()
            node = ('binop', op['value'], node, right)
        return node

    def parse_equality_expr(self):
        node = self.parse_comparison_expr()
        while self.match_op(['==','!=']):
            op = self.consume('OPERATOR')
            right = self.parse_comparison_expr()
            node = ('binop', op['value'], node, right)
        return node

    def parse_comparison_expr(self):
        node = self.parse_additive_expr()
        while self.match_op(['>','>=','<','<=']):
            op = self.consume('OPERATOR')
            right = self.parse_additive_expr()
            node = ('binop', op['value'], node, right)
        return node

    def parse_additive_expr(self):
        node = self.parse_multiplicative_expr()
        while self.match_op(['+','-']):
            op = self.consume('OPERATOR')
            right = self.parse_multiplicative_expr()
            node = ('binop', op['value'], node, right)
        return node

    def parse_multiplicative_expr(self):
        node = self.parse_unary_expr()
        while self.match_op(['*','/','%']):
            op = self.consume('OPERATOR')
            right = self.parse_unary_expr()
            node = ('binop', op['value'], node, right)
        return node

    def parse_unary_expr(self):
        if self.match_op(['-']):
            op = self.consume('OPERATOR')
            node = self.parse_unary_expr()
            return ('unaryop', op['value'], node)
        return self.parse_primary()

    def parse_primary(self):
        tok = self.current_token()
        if tok is None:
            raise ValueError("Unexpected end of input in primary")

        if tok['type'] == 'NUMBER':
            self.consume('NUMBER')
            return ('number', tok['value'])
        elif tok['type'] == 'STRING':
            self.consume('STRING')
            return ('string', tok['value'])
        elif tok['type'] == 'BOOLEAN':
            self.consume('BOOLEAN')
            return ('boolean', tok['value'])
        elif tok['type'] == 'LPAREN':
            self.consume('LPAREN')
            node = self.parse_expression()
            if not self.match('RPAREN'):
                raise ValueError("Missing closing parenthesis")
            self.consume('RPAREN')
            return node
        elif tok['type'] == 'IDENT':
            return self.parse_identifier_call()
        else:
            raise ValueError(f"Unexpected token in primary: {tok}")

    def parse_identifier_call(self):
        first = self.consume('IDENT')
        chain = [first['value']]

        while self.match('DOT'):
            self.consume('DOT')
            attr = self.consume('IDENT')
            chain.append(attr['value'])

        # Check if '(' follows for a function call
        if self.match('LPAREN'):
            self.consume('LPAREN')
            args = []
            # If immediate RPAREN -> no args
            if self.match('RPAREN'):
                self.consume('RPAREN')
                return ('call', chain, args)
            else:
                original_pos = self.pos
                try:
                    first_arg = self.parse_expression()
                    args.append(first_arg)
                    while self.match('COMMA'):
                        self.consume('COMMA')
                        args.append(self.parse_expression())
                    if not self.match('RPAREN'):
                        raise ValueError("Missing closing parenthesis in function call")
                    self.consume('RPAREN')
                except Exception:
                    # Fallback: consume until RPAREN to recover
                    args = []
                    while self.current_token() and self.current_token()['type'] != 'RPAREN':
                        self.pos += 1
                    if self.match('RPAREN'):
                        self.consume('RPAREN')
                return ('call', chain, args)
        else:
            return ('chain', chain)

def evaluate_ast(node, varmap):
    ntype = node[0]

    if ntype == 'number':
        return node[1]
    elif ntype == 'string':
        return node[1]
    elif ntype == 'boolean':
        return node[1]
    elif ntype == 'chain':
        return resolve_chain(node[1], varmap)
    elif ntype == 'call':
        chain = node[1]
        args_nodes = node[2]
        args = [evaluate_ast(a, varmap) for a in args_nodes]
        target = resolve_chain(chain, varmap)
        return call_function_or_method(target, args, varmap)
    elif ntype == 'binop':
        op = node[1]
        left = evaluate_ast(node[2], varmap)
        right = evaluate_ast(node[3], varmap)
        return eval_binop(op, left, right)
    elif ntype == 'unaryop':
        op = node[1]
        val = evaluate_ast(node[2], varmap)
        if op == '-':
            return -val
        else:
            raise ValueError(f"Unsupported unary operator: {op}")
    else:
        raise ValueError(f"Unknown AST node type: {ntype}")

def resolve_chain(chain, varmap):
    base = chain[0]
    if base in varmap:
        current = varmap[base]
    else:
        if base in FUNCTIONS or base in FUNCTIONS_USER:
            current = base
        else:
            raise ValueError(f"Undefined variable or function: {base}")

    for part in chain[1:]:
        if isinstance(current, dict) and '__class__' in current:
            class_name = current['__class__']
            class_def = CLASSES.get(class_name, None)
            if not class_def:
                raise ValueError(f"Undefined class '{class_name}'")

            if part in current:
                current = current[part]
            else:
                if part in class_def['methods']:
                    return ('class_method', class_name, current, part)
                else:
                    raise ValueError(f"'{part}' is not a property or method of instance of class '{class_name}'")
        else:
            if not hasattr(current, part):
                raise ValueError(f"Object of type '{type(current).__name__}' has no attribute '{part}'.")
            current = getattr(current, part)

    return current

def call_function_or_method(target, args, varmap):
    if isinstance(target, tuple) and target[0] == 'class_method':
        _, class_name, instance_obj, method_name = target
        class_def = CLASSES[class_name]
        method_body = class_def['methods'][method_name]
        method_varmap = varmap.copy()
        for k, v in instance_obj.items():
            method_varmap[k] = v
        return_flag, return_value = interpreter(method_body, method_varmap)
        for prop in class_def['properties']:
            if prop in method_varmap:
                instance_obj[prop] = method_varmap[prop]
        return return_value if return_flag else None

    if isinstance(target, str):
        if target in FUNCTIONS:
            func = FUNCTIONS[target]
            return func(*args)
        elif target in FUNCTIONS_USER:
            func_def = FUNCTIONS_USER[target]
            func_varmap = varmap.copy()
            params = func_def['params']
            if len(args) != len(params):
                raise ValueError(f"Function '{target}' expects {len(params)} arguments, got {len(args)}.")
            for p, a in zip(params, args):
                func_varmap[p] = a
            return_flag, return_value = interpreter(func_def['body'], func_varmap)
            return return_value if return_flag else None
        else:
            raise ValueError(f"Undefined function: {target}")
    else:
        if not callable(target):
            raise ValueError("Attempted to call a non-callable object.")
        return target(*args)

def eval_binop(op, left, right):
    try:
        if op == '+':
            # If either operand is a string, perform string concatenation
            if isinstance(left, str) or isinstance(right, str):
                return str(left) + str(right)
            return left + right
        elif op == '-':
            return left - right
        elif op == '*':
            return left * right
        elif op == '/':
            return left / right
        elif op == '%':
            return left % right
        elif op == '^':
            return left ** right
        elif op == '==':
            return left == right
        elif op == '!=':
            return left != right
        elif op == '<':
            return left < right
        elif op == '>':
            return left > right
        elif op == '<=':
            return left <= right
        elif op == '>=':
            return left >= right
        elif op == 'and':
            return bool(left and right)
        elif op == 'or':
            return bool(left or right)
        else:
            raise ValueError(f"Unsupported operator: {op}")
    except ZeroDivisionError:
        raise ValueError('Division by zero.')

def eval_expr(expression, varmap):
    expression = re.sub(r'\bnot\s+equal\s+to\b', '!=', expression)
    expression = re.sub(r'\bequals\b', '==', expression)
    tokens = tokenize(expression)
    parser = Parser(tokens)
    ast = parser.parse_expression()
    return evaluate_ast(ast, varmap)

def get_block(lines, start_index):
    block = []
    index = start_index + 1
    open_braces = 1
    while index < len(lines):
        line = lines[index].strip()
        if '{' in line:
            open_braces += line.count('{')
        if '}' in line:
            open_braces -= line.count('}')
            if open_braces == 0:
                break
        block.append(line)
        index += 1
    if open_braces != 0:
        print(f"Error: Block starting at line {start_index + 1} was never closed.")
        sys.exit(1)
    return block, index

def process_line(line, varmap, line_number):
    if line.startswith("list"):
        match = re.match(r'^list\s+(\w+)\s*=\s*\((.*?)\)$', line)
        if not match:
            print(f"Error on line {line_number}: Invalid list declaration syntax.")
            sys.exit(1)
        var_name = match.group(1)
        init_values = match.group(2).strip()
        if not init_values:
            list_obj = smpl_lists.List()
        else:
            try:
                values = [eval_expr(v.strip(), varmap) for v in init_values.split(',')]
            except Exception as e:
                print(f"Error on line {line_number}: Invalid list initialization. {e}")
                sys.exit(1)
            list_obj = smpl_lists.List(values)
        varmap[var_name] = list_obj
        return

    if line.startswith("print"):
        try:
            expression = re.search(r'\((.*?)\)', line).group(1).strip()
        except AttributeError:
            print(f"Error on line {line_number}: Invalid print statement syntax.")
            sys.exit(1)
        value = eval_expr(expression, varmap)
        print(value)
        return

    method_call_match = re.match(r'^(\w+)\.(\w+)\((.*?)\)$', line)
    if method_call_match:
        obj_name, method_name, arg_str = method_call_match.groups()
        if obj_name not in varmap:
            print(f"Error on line {line_number}: Undefined object '{obj_name}'.")
            sys.exit(1)
        obj = varmap[obj_name]
        args = []
        if arg_str.strip():
            args = [eval_expr(arg.strip(), varmap) for arg in arg_str.split(',')]
        if isinstance(obj, dict) and '__class__' in obj:
            class_name = obj['__class__']
            class_def = CLASSES.get(class_name, None)
            if not class_def:
                print(f"Error on line {line_number}: Undefined class '{class_name}'.")
                sys.exit(1)
            if method_name not in class_def['methods']:
                print(f"Error on line {line_number}: Undefined method '{method_name}' in class '{class_name}'.")
                sys.exit(1)
            target = ('class_method', class_name, obj, method_name)
        else:
            if isinstance(obj, smpl_lists.List):
                if hasattr(obj, method_name):
                    target = getattr(obj, method_name)
                else:
                    print(f"Error on line {line_number}: Undefined method '{method_name}' in List.")
                    sys.exit(1)
            else:
                print(f"Error on line {line_number}: '{obj_name}' is not an object with methods.")
                sys.exit(1)

        call_function_or_method(target, args, varmap)
        return

    object_prop_match = re.match(r'^(\w+)\.(\w+)\s*=\s*(.+)$', line)
    if object_prop_match:
        obj_name, prop_name, expr = object_prop_match.groups()
        if obj_name not in varmap:
            print(f"Error on line {line_number}: Undefined object '{obj_name}'.")
            sys.exit(1)
        obj = varmap[obj_name]
        if isinstance(obj, smpl_lists.List):
            print(f"Error on line {line_number}: Cannot directly assign to list items. Use 'set' method.")
            sys.exit(1)
        elif isinstance(obj, dict) and '__class__' in obj:
            value = eval_expr(expr.strip(), varmap)
            obj[prop_name] = value
            varmap[obj_name] = obj
        else:
            print(f"Error on line {line_number}: '{obj_name}' is not an object.")
            sys.exit(1)
        return

    var_assign_match = re.match(r'^(\w+)\s*=\s*(.+)$', line)
    if var_assign_match:
        var, expr = var_assign_match.groups()
        var = var.strip()
        expr = expr.strip().rstrip(';')

        if expr.startswith("new "):
            class_name = expr[4:].strip()
            if class_name == "List":
                list_obj = smpl_lists.List()
                varmap[var] = list_obj
            elif class_name in CLASSES:
                class_def = CLASSES[class_name]
                obj = {'__class__': class_name}
                for prop, default in class_def['properties'].items():
                    obj[prop] = default
                varmap[var] = obj
            else:
                print(f"Error on line {line_number}: Undefined class '{class_name}'.")
                sys.exit(1)
            return

        list_declaration_match = re.match(r'^list\s+(\w+)\s*=\s*\((.*?)\)$', expr)
        if list_declaration_match:
            var_name, init_values = list_declaration_match.groups()
            if not init_values.strip():
                list_obj = smpl_lists.List()
            else:
                try:
                    values = [eval_expr(v.strip(), varmap) for v in init_values.split(',')]
                except Exception as e:
                    print(f"Error on line {line_number}: Invalid list initialization. {e}")
                    sys.exit(1)
                list_obj = smpl_lists.List(values)
            varmap[var_name] = list_obj
            return

        varmap[var] = eval_expr(expr, varmap)
        return

    print(f"Error on line {line_number}: Undefined statement '{line}'.")
    sys.exit(1)

def interpreter(lines, varmap, start_index=0):
    index = start_index
    while index < len(lines):
        line = lines[index].strip()
        current_line_number = index + 1

        if not line or line.startswith('//') or line.startswith('#'):
            index += 1
            continue

        if line.startswith('class'):
            match = re.match(r'^class\s+(\w+)\s*\{', line)
            if not match:
                print(f"Error on line {current_line_number}: Invalid class definition syntax.")
                sys.exit(1)
            class_name = match.group(1)
            class_body, block_end_index = get_block(lines, index)
            class_properties = {}
            class_methods = {}
            i = 0
            while i < len(class_body):
                class_line = class_body[i].strip()
                if not class_line or class_line.startswith('//') or class_line.startswith('#'):
                    i += 1
                    continue
                if class_line.startswith('func'):
                    method_match = re.match(r'^func\s+(\w+)\s*\{', class_line)
                    if not method_match:
                        print(f"Error on line {current_line_number}: Invalid method definition syntax in class '{class_name}'.")
                        sys.exit(1)
                    method_name = method_match.group(1)
                    method_body, method_block_end = get_block(class_body, i)
                    class_methods[method_name] = method_body
                    i = method_block_end + 1
                else:
                    prop_match = re.match(r'^(\w+)\s*=\s*(.+)$', class_line)
                    if not prop_match:
                        print(f"Error on line {current_line_number}: Invalid property definition in class '{class_name}'.")
                        sys.exit(1)
                    prop_name, prop_value = prop_match.groups()
                    prop_value = prop_value.strip().rstrip(';')
                    if prop_value.startswith('"') and prop_value.endswith('"'):
                        class_properties[prop_name] = prop_value[1:-1]
                    elif prop_value.isdigit():
                        class_properties[prop_name] = int(prop_value)
                    elif re.match(r'^\d+\.\d+$', prop_value):
                        class_properties[prop_name] = float(prop_value)
                    elif prop_value in ['true', 'false']:
                        class_properties[prop_name] = True if prop_value == 'true' else False
                    else:
                        class_properties[prop_name] = prop_value
                    i += 1
            CLASSES[class_name] = {
                'properties': class_properties,
                'methods': class_methods
            }
            index = block_end_index + 1
            continue

        if line.startswith('func'):
            match = re.match(r'^func\s+(\w+)(?:\s+using\s+([\w\s,]+))?\s*\{', line)
            if not match:
                print(f"Error on line {current_line_number}: Invalid function definition syntax.")
                sys.exit(1)
            func_name = match.group(1)
            params_str = match.group(2)
            params = [param.strip() for param in params_str.split(',')] if params_str else []
            func_body, block_end_index = get_block(lines, index)
            FUNCTIONS_USER[func_name] = {'params': params, 'body': func_body}
            index = block_end_index + 1
            continue

        if line.startswith('return'):
            return_expr = line[len('return'):].strip()
            return_value = eval_expr(return_expr, varmap)
            return True, return_value

        if line.startswith('if'):
            conditions = []
            blocks = []
            match = re.match(r'^if\s+(.+?)\s*\{', line)
            if match:
                condition = match.group(1).strip()
            else:
                print(f"Error on line {current_line_number}: Invalid if statement syntax.")
                sys.exit(1)
            condition_result = eval_expr(condition, varmap)
            block, block_end_index = get_block(lines, index)
            conditions.append(condition_result)
            blocks.append(block)
            index = block_end_index + 1
            while index < len(lines):
                next_line = lines[index].strip()
                if next_line.startswith('elif'):
                    match = re.match(r'^elif\s+(.+?)\s*\{', next_line)
                    if match:
                        elif_condition = match.group(1).strip()
                    else:
                        print(f"Error on line {index + 1}: Invalid elif statement syntax.")
                        sys.exit(1)
                    elif_condition_result = eval_expr(elif_condition, varmap)
                    elif_block, elif_block_end = get_block(lines, index)
                    conditions.append(elif_condition_result)
                    blocks.append(elif_block)
                    index = elif_block_end + 1
                elif next_line.startswith('else'):
                    match = re.match(r'^else\s*\{', next_line)
                    if match:
                        else_block, else_block_end = get_block(lines, index)
                        conditions.append(True)
                        blocks.append(else_block)
                        index = else_block_end + 1
                        break
                    else:
                        print(f"Error on line {index + 1}: Invalid else statement syntax.")
                        sys.exit(1)
                else:
                    break
            for cond, blk in zip(conditions, blocks):
                if cond:
                    return_flag, return_value = interpreter(blk, varmap)
                    if return_flag:
                        return (True, return_value)
                    break
            continue

        if line.startswith('elif') or line.startswith('else'):
            print(f"Error on line {current_line_number}: '{line.split()[0]}' without preceding 'if'.")
            sys.exit(1)

        if line.startswith('while'):
            try:
                match = re.match(r'while\s+(.+)', line)
                if match:
                    condition_part = match.group(1).strip()
                    if condition_part.endswith('{'):
                        condition = condition_part[:-1].strip()
                    else:
                        condition = condition_part
                else:
                    raise AttributeError
            except AttributeError:
                print(f"Error on line {current_line_number}: Invalid while statement syntax.")
                sys.exit(1)

            block, block_end_index = get_block(lines, index)
            while eval_expr(condition, varmap):
                return_flag, return_value = interpreter(block, varmap)
                if return_flag:
                    return (True, return_value)
            index = block_end_index + 1
            continue

        if line.startswith('for'):
            try:
                match = re.match(r'for\s+(\w+)\s+from\s+(.*?)\s+to\s+(.*)', line)
                if not match:
                    raise ValueError
                var = match.group(1)
                start_expr = match.group(2).strip()
                end_expr = match.group(3).strip()
                if end_expr.endswith('{'):
                    end_expr = end_expr[:-1].strip()
            except ValueError:
                print(f"Error on line {current_line_number}: Invalid for loop syntax.")
                sys.exit(1)

            try:
                start = eval_expr(start_expr, varmap)
                end = eval_expr(end_expr, varmap) + 1
            except Exception as e:
                print(f"Error on line {current_line_number}: Invalid for loop range. {e}")
                sys.exit(1)

            block, block_end_index = get_block(lines, index)
            for i in range(start, end):
                varmap[var] = i
                return_flag, return_value = interpreter(block, varmap)
                if return_flag:
                    return (True, return_value)
            index = block_end_index + 1
            continue

        if (line.startswith('print') or
            re.match(r'^\w+\.\w+\s*=\s*.+$', line) or
            re.match(r'^\w+\.\w+\(.*\)$', line) or
            '=' in line or
            line.startswith("list")):
            process_line(line, varmap, current_line_number)
            index += 1
            continue

        print(f"Error on line {current_line_number}: Undefined statement '{line}'.")
        sys.exit(1)

    return False, None

def main(filename):
    varmap = {}
    varmap.update(FUNCTIONS)

    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    interpreter(lines, varmap)

def cli():
    import sys
    if len(sys.argv) < 2:
        print("Usage: smpl <filename.smpl>")
        sys.exit(1)
    filename = sys.argv[1]
    main(filename)