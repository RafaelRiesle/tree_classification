class OldDisturbancePruner:
    def __init__(self, on=True):
        self.on = on

    def delete_old_disturbances(self, data):
        filtered_data = [
            row for row in data
            if not (row.get('disturbance_year', 0) < 2017 and row.get('species') == 'disturbed')
        ]
        return filtered_data

    def run(self, data):
        if self.on:
            return self.delete_old_disturbances(data)
        return data
