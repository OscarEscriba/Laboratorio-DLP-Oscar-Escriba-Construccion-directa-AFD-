# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import matplotlib.patches as patches
import numpy as np
import math
import csv 

class Node:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
        self.nullable = None
        self.firstpos = set()
        self.lastpos = set()
        self.position = None
        
class RegexToDFA:
    def __init__(self, regex):
        self.original_regex = regex
        self.augmented_regex = f"({regex})#"
        self.alphabet = set([char for char in self.original_regex 
                            if char not in '()*|+?·#'])
        self.pos_to_symbol = {}  # Mapeo de posiciones a símbolos
        self.next_pos = 1
        self.followpos = defaultdict(set)
        self.syntax_tree = None
        self.dfa = None
        self.minimized_dfa = None
        
    def parse_regex(self):
        # Primero convertimos la expresión a postfix usando Shunting Yard
        postfix = self.infix_to_postfix(self.augmented_regex)
        # Luego construimos el árbol sintáctico
        self.syntax_tree = self.build_syntax_tree(postfix)
        # Calculamos nullable, firstpos, lastpos y followpos
        self.calculate_tree_properties(self.syntax_tree)
        # Calculamos followpos
        self.calculate_followpos(self.syntax_tree)
        
    def infix_to_postfix(self, regex):
        # Implementación del algoritmo Shunting Yard para convertir infix a postfix
        output = []
        stack = []
        
        # Agregar concatenación explícita
        explicit_regex = []
        for i in range(len(regex) - 1):
            explicit_regex.append(regex[i])
            if regex[i] not in '(|' and regex[i+1] not in ')|*+?':
                explicit_regex.append('·')
        explicit_regex.append(regex[-1])
        regex = ''.join(explicit_regex)
        
        # Definir precedencia
        precedence = {'|': 1, '·': 2, '*': 3, '+': 3, '?': 3}
        
        for char in regex:
            if char not in '()|*+?·':
                output.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if stack and stack[-1] == '(':
                    stack.pop()  # Descartar el paréntesis izquierdo
            else:  # Operador
                while (stack and stack[-1] != '(' and 
                       precedence.get(stack[-1], 0) >= precedence.get(char, 0)):
                    output.append(stack.pop())
                stack.append(char)
        
        # Vaciar la pila
        while stack:
            output.append(stack.pop())
            
        return ''.join(output)
    
    def build_syntax_tree(self, postfix):
        stack = []
        
        for char in postfix:
            if char in '*+?':
                # Operadores unarios
                if not stack:
                    raise ValueError(f"Expresión inválida: {postfix}")
                    
                node = Node(char)
                node.left = stack.pop()
                stack.append(node)
                
            elif char in '|·':
                # Operadores binarios
                if len(stack) < 2:
                    raise ValueError(f"Expresión inválida: {postfix}")
                    
                node = Node(char)
                node.right = stack.pop()
                node.left = stack.pop()
                stack.append(node)
                
            else:
                # Símbolo del alfabeto o #
                node = Node(char)
                if char != 'ε':  # Si no es epsilon, le asignamos una posición
                    node.position = self.next_pos
                    self.pos_to_symbol[self.next_pos] = char
                    self.next_pos += 1
                stack.append(node)
        
        if len(stack) != 1:
            raise ValueError(f"Expresión inválida: {postfix}")
            
        return stack[0]
    
    def calculate_tree_properties(self, node):
        if not node:
            return
        
        # Procesar nodos hijos primero (postorden)
        self.calculate_tree_properties(node.left)
        self.calculate_tree_properties(node.right)
        
        # Calcular Nullable
        if node.data == 'ε':
            node.nullable = True
        elif node.data in self.alphabet or node.data == '#':
            node.nullable = False
        elif node.data == '|':
            node.nullable = node.left.nullable or node.right.nullable
        elif node.data == '·':
            node.nullable = node.left.nullable and node.right.nullable
        elif node.data in '*?':
            node.nullable = True
        elif node.data == '+':
            node.nullable = node.left.nullable
        
        # Calcular Firstpos
        if node.data == 'ε':
            node.firstpos = set()
        elif node.data in self.alphabet or node.data == '#':
            node.firstpos = {node.position}
        elif node.data == '|':
            node.firstpos = node.left.firstpos.union(node.right.firstpos)
        elif node.data == '·':
            if node.left.nullable:
                node.firstpos = node.left.firstpos.union(node.right.firstpos)
            else:
                node.firstpos = node.left.firstpos
        elif node.data in '*+?':
            node.firstpos = node.left.firstpos
        
        # Calcular Lastpos
        if node.data == 'ε':
            node.lastpos = set()
        elif node.data in self.alphabet or node.data == '#':
            node.lastpos = {node.position}
        elif node.data == '|':
            node.lastpos = node.left.lastpos.union(node.right.lastpos)
        elif node.data == '·':
            if node.right.nullable:
                node.lastpos = node.left.lastpos.union(node.right.lastpos)
            else:
                node.lastpos = node.right.lastpos
        elif node.data in '*+?':
            node.lastpos = node.left.lastpos
    
    def calculate_followpos(self, node):
        if not node:
            return
        
        # Caso 1: Para nodo con operador concat '·'
        if node.data == '·':
            for i in node.left.lastpos:
                self.followpos[i] = self.followpos[i].union(node.right.firstpos)
        
        # Caso 2: Para nodo con operador '*'
        if node.data in '*+':
            for i in node.lastpos:
                self.followpos[i] = self.followpos[i].union(node.firstpos)
        
        # Procesar recursivamente los nodos hijo
        self.calculate_followpos(node.left)
        self.calculate_followpos(node.right)
    
    def construct_dfa(self):
        # Estado inicial: firstpos de la raíz
        initial_state = frozenset(self.syntax_tree.firstpos)
        
        # Inicializar DFA
        states = {initial_state: 0}  # Mapeo de conjuntos de posiciones a estados
        unmarked_states = [initial_state]
        transitions = {}
        final_states = set()
        
        # Encontrar posición del símbolo de fin '#'
        end_pos = None
        for pos, symbol in self.pos_to_symbol.items():
            if symbol == '#':
                end_pos = pos
                break
        
        if end_pos is None:
            raise ValueError("No se encontró el símbolo de fin '#'")
        
        # Construir DFA usando el algoritmo de construcción directa
        while unmarked_states:
            current_state = unmarked_states.pop(0)
            
            # Si el estado contiene la posición del símbolo de fin, es un estado final
            if end_pos in current_state:
                final_states.add(states[current_state])
            
            # Para cada símbolo en el alfabeto (excepto ε)
            for symbol in self.alphabet:
                # Calcular el Move(estado, símbolo)
                next_state_positions = set()
                for pos in current_state:
                    if self.pos_to_symbol.get(pos) == symbol:
                        next_state_positions = next_state_positions.union(self.followpos[pos])
                
                if not next_state_positions:
                    continue
                
                next_state = frozenset(next_state_positions)
                
                # Si es un nuevo estado, agregarlo a los estados no marcados
                if next_state not in states:
                    states[next_state] = len(states)
                    unmarked_states.append(next_state)
                
                # Añadir la transición
                src = states[current_state]
                dest = states[next_state]
                transitions[(src, symbol)] = dest
        
        # Almacenar el DFA
        self.dfa = {
            'states': len(states),
            'initial': 0,
            'final_states': final_states,
            'transitions': transitions
        }
        
        return self.dfa
    
    def minimize_dfa(self):
        if not self.dfa:
            raise ValueError("Primero debe construir el DFA")
        
        # Algoritmo de Hopcroft para minimización de DFA
        transitions = self.dfa['transitions']
        final_states = self.dfa['final_states']
        num_states = self.dfa['states']
        
        # Inicializar particiones: estados finales y no finales
        partitions = [
            set(range(num_states)) - final_states,  # Estados no finales
            final_states  # Estados finales
        ]
        # Eliminar particiones vacías
        partitions = [p for p in partitions if p]
        
        workset = []
        for partition in partitions:
            if partition:
                workset.append(partition)
        
        while workset:
            current_set = workset.pop(0)
            
            for symbol in self.alphabet:
                # Encontrar estados que llevan a current_set con el símbolo
                predecessors = set()
                for state in range(num_states):
                    if (state, symbol) in transitions and transitions[(state, symbol)] in current_set:
                        predecessors.add(state)
                
                # Actualizar particiones
                new_partitions = []
                for partition in partitions:
                    intersection = partition.intersection(predecessors)
                    difference = partition - predecessors
                    
                    if intersection and difference:
                        new_partitions.append(intersection)
                        new_partitions.append(difference)
                        
                        # Actualizar workset
                        if partition in workset:
                            workset.remove(partition)
                            workset.append(intersection)
                            workset.append(difference)
                        else:
                            if len(intersection) <= len(difference):
                                workset.append(intersection)
                            else:
                                workset.append(difference)
                    else:
                        new_partitions.append(partition)
                
                partitions = new_partitions
        
        # Construir el DFA minimizado
        # Mapear estados originales a nuevos estados
        state_mapping = {}
        for i, partition in enumerate(partitions):
            for state in partition:
                state_mapping[state] = i
        
        # Construir nuevas transiciones
        new_transitions = {}
        for (src, symbol), dest in transitions.items():
            new_src = state_mapping[src]
            new_dest = state_mapping[dest]
            new_transitions[(new_src, symbol)] = new_dest
        
        # Encontrar nuevos estados finales
        new_final_states = set()
        for final_state in final_states:
            new_final_states.add(state_mapping[final_state])
        
        # Encontrar nuevo estado inicial
        new_initial = state_mapping[self.dfa['initial']]
        
        # Almacenar el DFA minimizado
        self.minimized_dfa = {
            'states': len(partitions),
            'initial': new_initial,
            'final_states': new_final_states,
            'transitions': new_transitions
        }
        
        return self.minimized_dfa
    
    def simulate_dfa(self, input_string, minimized=True):
        if minimized and self.minimized_dfa:
            dfa = self.minimized_dfa
        elif self.dfa:
            dfa = self.dfa
        else:
            raise ValueError("Primero debe construir el DFA")
        
        current_state = dfa['initial']
        
        for char in input_string:
            if char not in self.alphabet:
                return False  # Carácter no reconocido
            
            if (current_state, char) in dfa['transitions']:
                current_state = dfa['transitions'][(current_state, char)]
            else:
                return False  # No hay transición para este carácter
        
        return current_state in dfa['final_states']
    
    def visualize_automaton(self, minimized=True, filename="automaton"):
        if minimized and self.minimized_dfa:
            dfa = self.minimized_dfa
            title = "Minimized DFA"
        elif self.dfa:
            dfa = self.dfa
            title = "Direct Construction DFA"
        else:
            raise ValueError("Primero debe construir el DFA")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Obtener el número de estados y calcular sus posiciones en círculo
        num_states = dfa['states']
        radius = 4
        state_positions = {}
        
        for i in range(num_states):
            angle = 2 * math.pi * i / num_states
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            state_positions[i] = (x, y)
        
        # Dibujar estados
        for state, (x, y) in state_positions.items():
            # Determinar el estilo según el tipo de estado
            if state == dfa['initial']:
                circle = patches.Circle((x, y), 0.5, edgecolor='blue', facecolor='lightblue', linewidth=2)
            elif state in dfa['final_states']:
                # Dibujar el círculo exterior
                outer_circle = patches.Circle((x, y), 0.6, edgecolor='black', facecolor='none')
                ax.add_patch(outer_circle)
                circle = patches.Circle((x, y), 0.5, edgecolor='red', facecolor='lightpink')
            else:
                circle = patches.Circle((x, y), 0.5, edgecolor='black', facecolor='none')
            
            ax.add_patch(circle)
            
            # Añadir etiqueta del estado
            ax.text(x, y, str(state), ha='center', va='center')
        
        # Dibujar flecha inicial
        if dfa['initial'] in state_positions:
            init_x, init_y = state_positions[dfa['initial']]
            ax.arrow(init_x - 1.5, init_y, 0.7, 0, head_width=0.2, head_length=0.2, fc='green', ec='green')
        
        # Dibujar transiciones
        for (src, symbol), dest in dfa['transitions'].items():
            # Obtener posiciones
            if src in state_positions and dest in state_positions:
                src_x, src_y = state_positions[src]
                dest_x, dest_y = state_positions[dest]
                
                # Para auto-transiciones
                if src == dest:
                    # Dibujar un arco sobre el estado
                    loop = patches.Arc((src_x, src_y + 0.7), 0.8, 0.8, theta1=180, theta2=360, edgecolor='gray')
                    ax.add_patch(loop)
                    # Añadir la etiqueta
                    ax.text(src_x, src_y + 1.1, symbol, ha='center', va='center', color='darkblue')
                else:
                    # Calcular la dirección normalizada
                    dx, dy = dest_x - src_x, dest_y - src_y
                    length = math.sqrt(dx*dx + dy*dy)
                    dx, dy = dx/length, dy/length
                    
                    # Calcular puntos de inicio y fin (ajustados para evitar solapamiento con círculos)
                    start_x = src_x + dx * 0.5
                    start_y = src_y + dy * 0.5
                    end_x = dest_x - dx * 0.5
                    end_y = dest_y - dy * 0.5
                    
                    # Crear flecha
                    ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y, 
                             head_width=0.2, head_length=0.2, fc='black', ec='black',
                             length_includes_head=True)
                    
                    # Añadir etiqueta en el medio
                    mid_x = (start_x + end_x) / 2 + dy * 0.2  # Desplazar perpendicular para evitar solapamiento
                    mid_y = (start_y + end_y) / 2 - dx * 0.2
                    ax.text(mid_x, mid_y, symbol, ha='center', va='center', color='darkblue',
                            bbox=dict(facecolor='white', edgecolor='none', pad=1))
        
        # Configurar el gráfico
        ax.set_aspect('equal')
        ax.set_xlim(-radius-2, radius+2)
        ax.set_ylim(-radius-2, radius+2)
        ax.axis('off')
        ax.set_title(title)
        
        # Guardar el gráfico
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
        print(f"Automaton visualization saved as '{filename}.png'")
        
        # Mostrar el gráfico
        plt.show()
        
    def process(self):
        self.parse_regex()
        self.construct_dfa()
        self.minimize_dfa()
    
    #Parte para imprimir en el csv la descripcion de mi afd DEEPTSEK
    def export_dfa(self, filename="dfa_description.csv", minimized=True):
        if minimized and self.minimized_dfa:
            dfa = self.minimized_dfa
        elif self.dfa:
            dfa = self.dfa
        else:
            raise ValueError("Primero debe construir el DFA")
        
        # Obtener la información del AFD
        states = list(range(dfa['states']))
        initial_state = dfa['initial']
        final_states = list(dfa['final_states'])
        transitions = dfa['transitions']
        
        # Crear una lista para almacenar las transiciones
        transitions_list = []
        for (src, symbol), dest in transitions.items():
            transitions_list.append({
                'source': src,
                'symbol': symbol,
                'destination': dest
            })
        
        # Si se proporciona un nombre de archivo, exportar a CSV
        if filename:
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                # Escribir los estados
                file.write("Estados:\n")
                file.write(",".join(map(str, states)))  # Convertir estados a cadena
                file.write("\n\n")
                
                # Escribir el estado inicial
                file.write("Estado Inicial:\n")
                file.write(str(initial_state))
                file.write("\n\n")
                
                # Escribir los estados finales
                file.write("Estados Finales:\n")
                file.write(",".join(map(str, final_states)))  # Convertir estados finales a cadena
                file.write("\n\n")
                
                # Escribir las transiciones
                file.write("Transiciones:\n")
                writer = csv.DictWriter(file, fieldnames=['source', 'symbol', 'destination'])
                writer.writeheader()  # Escribir la cabecera
                writer.writerows(transitions_list)  # Escribir todas las transiciones
                    
            print(f"Descripción del AFD guardada en '{filename}'")
            
def main():
    while True:
        print("\n==== Analizador de Expresiones Regulares ====")
        regex = input("Ingrese una expresión regular (o 'salir' para terminar): ")
        
        if regex.lower() == 'salir':
            break
        
        try:
            # Construir el AFD a partir de la expresión regular
            converter = RegexToDFA(regex)
            converter.process()
            
            # Visualizar el AFD original
            converter.visualize_automaton(minimized=False, filename="original_dfa")
            
            # Visualizar el AFD minimizado
            converter.visualize_automaton(minimized=True, filename="minimized_dfa")
            
            # Exportar la descripción del AFD a un archivo CSV
            converter.export_dfa(filename="dfa_description.csv", minimized=True)
            
            # Procesar cadenas
            while True:
                w = input("\nIngrese una cadena para verificar (o 'volver' para nueva regex): ")
                
                if w.lower() == 'volver':
                    break
                
                # Verificar si la cadena es aceptada
                is_accepted = converter.simulate_dfa(w)
                
                if is_accepted:
                    print(f"La cadena '{w}' es ACEPTADA por el autómata.")
                else:
                    print(f"La cadena '{w}' NO es aceptada por el autómata.")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()