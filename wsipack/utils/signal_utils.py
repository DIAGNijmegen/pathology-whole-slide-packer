import os, sys, signal, time
from pathlib import Path

class ExitHandler(object):
    _instance = None

    @staticmethod
    def instance():
        if ExitHandler._instance == None:
            ExitHandler()
        return ExitHandler._instance

    def __init__(self):
        if ExitHandler._instance != None:
            raise Exception("This class is a singleton!")
        else:
            self.handlers = []
            ExitHandler._instance = self
            signal.signal(signal.SIGINT, self.exit_gracefully)
            signal.signal(signal.SIGTERM, self.exit_gracefully)
            # signal.signal(signal.SIGKILL, self.exit_gracefully) #cant work, the process is terminated immediately

    def exit_gracefully(self, *args, **kwargs):
        # print(args, kwargs)
        print('exiting gracefully with %d handlers' % len(self.handlers))
        for handler in self.handlers:
            try:
                handler()
            except:
                print('Exception with exit handler %s' % str(handler))
                print(sys.exc_info())

    def add(self, handler):
        self.handlers.append(handler)

    def add_path_unlinker(self, path):
        fr = PathUnlinker(path)
        self.handlers.append(fr)
        return fr

    def remove(self, handler):
        if handler in self.handlers:
            self.handlers.remove(handler)
        else:
            print('handler %s not in exit handlers' % str(handler))


    def __str__(self):
        print('ExitHandler with %d handlers' % len(self.handlers))


class PathUnlinker(object):
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, *args, **kwargs):
        if self.path.exists():
            self.path.unlink()
            print('%s unlinked' % str(self.path))
        else:
            print("can't unlink non-existing %s" % str(self.path))

    def __str__(self):
        print('FileRemover for %s' % str(self.path))

class ExitPrinter(object):
    def __call__(self, *args, **kwargs):
        print('exit!')

if __name__ == '__main__':
    ExitHandler.instance().add(ExitPrinter())
    time.sleep(10)