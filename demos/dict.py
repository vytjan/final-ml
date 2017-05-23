import numpy as np
import difflib
import codecs

#######   training part    ###############
global samples
global responses
global dictionary
# samples = np.loadtxt('generalsamplesA.data',np.float32)
# responses = np.loadtxt('generalresponsesA.data',np.float32)

# dictionary = np.loadtxt('dictionary.dat', delimiter="\n")
# dictionary = np.loadtxt('dictionary.dat', converters={0: lambda x: unicode(x, 'utf-8')}, dtype='U2', delimiter="\n")
dictionary = codecs.open("dictionary.dat", encoding="utf-8").read().splitlines()
# s = codecs.open("dictionary.dat", encoding="utf-8")
# arr = numpy.frombuffer(s.replace("\n", ""), dtype="<U2")
# print(arr)
# dictionary = np.loadtxt('dictionary.dat', delimiter="\n")
print("".join(dictionary))
print(len(dictionary))

something = difflib.get_close_matches('o6ia', dictionary)
print("zodis yra ",something)