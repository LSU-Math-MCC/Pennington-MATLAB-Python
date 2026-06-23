'''
PointPairing.py
created by: Alex Mensen-Johnson
hardcoded point pattern based off of Escowell template marker files, pairs symmetrical points together for the selection
process
'''
class PointPicker:
    def __init__(self,points = None):
        x = 0
        if points is None:
            raise Exception('no points passed')
        for each in points:
            if isinstance(each,int):
                pass
            else:
                raise Exception("Passed List contains object of type {",type(each),"}, can only contain type int")
        self.points = points
        self.pointDict = {2: 3, 5: 7, 6: 8, 10: 12, 13: 14, 16: 18, 17: 19, 20: 22, 21: 23, 26: 27, 29: 41, 30: 42,
                          31: 43, 32: 44, 33: 45, 34: 46, 35: 47, 36: 48, 37: 49, 38: 50, 39: 51, 40: 52, 53: 63,
                          54: 64, 55 : 65, 56: 66, 57: 67, 58: 68, 59: 69, 60: 70, 61: 71, 62: 72}
        self.newList = []
        self._listbuilder()

    def _listbuilder(self):
        for key, value in self.pointDict.items():
            if key in self.points:
                if key not in self.newList:
                    self.newList.append(key)
                if value not in self.newList:
                    self.newList.append(value)
            if value in self.points:
                if key not in self.newList:
                    self.newList.append(key)
                if value not in self.newList:
                    self.newList.append(value)
        self.newList.sort()

    def getter(self):
        return self.newList

def __main__():
    exe = [2, 18, 29, 41, 60, 72]
    pp = PointPicker(exe)
    print(pp.getter())

if __name__ == '__main__':
    __main__()