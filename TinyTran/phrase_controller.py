import random
import collections

class PhraseController:
    def __init__(self, transition_matrix, start_state="Short"):
        self.states = ["Short", "Medium", "Long", "Silent"]
        self.matrix = self._validate_and_normalize(transition_matrix)
        self.current_state = start_state if start_state in self.states else "Short"

    def _validate_and_normalize(self, matrix):
        # Validate keys and normalize rows to sum to 1.0
        normalized_matrix = {}
        for state in self.states:
            if state not in matrix:
                raise ValueError(f"Missing state '{state}' in transition matrix.")
            row = matrix[state]
            if len(row) != len(self.states):
                raise ValueError(f"Transition row for '{state}' must have {len(self.states)} values.")
            row_sum = sum(row)
            if row_sum == 0:
                normalized_row = [0.0] * len(self.states)
            else:
                normalized_row = [round(value / row_sum, 4) for value in row]
            normalized_matrix[state] = normalized_row
        return normalized_matrix

    def next_state(self):
        weights = self.matrix[self.current_state]
        next_state = random.choices(self.states, weights=weights)[0]
        self.current_state = next_state
        return next_state

    def generate_sequence(self, length):
        sequence = []
        for _ in range(length):
            sequence.append(self.next_state())
        return sequence

    def print_sequence_stats(self, sequence):
        print("\nGenerated Sequence:")
        print(sequence)

        print("\nPhrase Counts:")
        counts = collections.Counter(sequence)
        for state in self.states:
            print(f"{state}: {counts.get(state, 0)}")

        print("\nTransition Matrix from Generated Sequence:")
        self._print_transition_matrix_from_sequence(sequence)

    def _print_transition_matrix_from_sequence(self, sequence):
        transitions = collections.Counter()
        total_transitions = {state: 0 for state in self.states}

        for i in range(len(sequence) - 1):
            from_state, to_state = sequence[i], sequence[i + 1]
            transitions[(from_state, to_state)] += 1
            total_transitions[from_state] += 1

        # Header
        header = f"| From \\ To | {' | '.join([f'{s:^8}' for s in self.states])} |"
        separator = "-" * len(header)
        print(header)
        print(separator)

        for from_state in self.states:
            row = f"| **{from_state:<6}** |"
            for to_state in self.states:
                count = transitions.get((from_state, to_state), 0)
                ratio = (count / total_transitions[from_state]) if total_transitions[from_state] else 0.0
                row += f" {count:^5} | {ratio:.2f} |"
            print(row)
        print()

# === Example Usage ===
if __name__ == "__main__":
    transition_matrix = {
        "Short": [0.59, 0.12, 0.15, 0.14],
        "Medium": [0.42, 0.15, 0.25, 0.18],
        "Long": [0.32, 0.16, 0.34, 0.18],
        "Silent": [0.36, 0.09, 0.13, 0.42]
    }

    controller = PhraseController(transition_matrix)
    generated_sequence = controller.generate_sequence(100)
    controller.print_sequence_stats(generated_sequence)

