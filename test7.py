from meliae import scanner
from meliae import loader
scanner.dump_all_objects('/home/wangq/dump.txt')
om = loader.load('/home/wangq/dump.txt')
om.compute_parents()
om.collapse_instance_dicts()
om.summarize()