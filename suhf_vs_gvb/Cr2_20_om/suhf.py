from pyphf import suscf, guess
from pyscf import lib

lib.num_threads(8)
xyz = '''Cr 0.0 0.0 0.0; Cr 0.0 0.0 2.0'''

mf1 = guess.from_frag(xyz, 'def2-tzvp', [[0],[1]], [0,0], [6,-6], cycle=50)
mf1 = guess.check_stab(mf1)
mf2 = suscf.SUHF(mf1)
mf2.max_cycle = 100
mf2.level_shift = 0.5
mf2.tofch = True
mf2.oldfch = 'Cr2_20_uhf.fch'
mf2.kernel()
