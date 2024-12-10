# smpl_lists.py

class List:
    def __init__(self, initial=None):
        if initial is None:
            self.items = []
        else:
            self.items = list(initial)

    def append(self, item):
        self.items.append(item)

    def get(self, index):
        try:
            return self.items[index]
        except IndexError:
            raise ValueError(f'Index {index} out of range.')

    def size(self):
        return len(self.items)

    def remove(self, item):
        try:
            self.items.remove(item)
        except ValueError:
            raise ValueError(f'Item {item} not found in list.')

    def __str__(self):
        return str(self.items)
