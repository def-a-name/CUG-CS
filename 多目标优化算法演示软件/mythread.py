import time
from manager import Manager
from PySide2.QtCore import QThread, QMutex, Signal, QObject
from PySide2.QtWidgets import QMessageBox
from threading import Thread


class my_Thread_run(QThread):
    finished = Signal(bool)

    def __init__(self, manager):
        super(my_Thread_run, self).__init__()
        self.m = manager

    def run(self):
        if(self.m.runAlgorithm()):
            # time.sleep(0.5)
            self.finished.emit(1)
        else:
            self.finished.emit(0)

# class mySignal(QObject):
#     intSignal = Signal(bool)
# class my_Thread_run(Thread):
#     finished = mySignal()

#     def __init__(self, manager):
#         super(my_Thread_run, self).__init__()
#         self.m = manager

#     def run(self):
#         if(self.m.runAlgorithm()):
#             # time.sleep(0.5)
#             self.finished.intSignal.emit(1)
#         else:
#             self.finished.intSignal.emit(0)