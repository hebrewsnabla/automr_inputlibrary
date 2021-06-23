from pyscf import gto, scf
from fch2py import fch2py
from ortho import check_orthonormal
from pyscf import mcscf, lib
from py2fch import py2fch
from shutil import copyfile
import numpy as np

lib.num_threads(1)

mol = gto.M()
# 2 atom(s)
mol.atom = '''
C1              0.00000000        0.00000000        0.00000000
C2              0.00000000        0.00000000        1.24000000
'''

mol.basis = {
'C1': gto.basis.parse('''
C     S
    0.666500000E+04    0.691583963E-03
    0.100000000E+04    0.532579615E-02
    0.228000000E+03    0.270607210E-01
    0.647100000E+02    0.101656846E+00
    0.210600000E+02    0.274574824E+00
    0.749500000E+01    0.448294319E+00
    0.279700000E+01    0.284902611E+00
    0.521500000E+00    0.151948592E-01
C     S
    0.666500000E+04   -0.293269653E-03
    0.100000000E+04   -0.231803547E-02
    0.228000000E+03   -0.114997860E-01
    0.647100000E+02   -0.468267270E-01
    0.210600000E+02   -0.128466169E+00
    0.749500000E+01   -0.301266272E+00
    0.279700000E+01   -0.255630702E+00
    0.521500000E+00    0.109379336E+01
C     S
    0.159600000E+00    0.100000000E+01
C     P
    0.943900000E+01    0.569792516E-01
    0.200200000E+01    0.313207212E+00
    0.545600000E+00    0.760376742E+00
C     P
    0.151700000E+00    0.100000000E+01
C     D
    0.550000000E+00    0.100000000E+01
'''),
'C2': gto.basis.parse('''
C     S
    0.666500000E+04    0.691583963E-03
    0.100000000E+04    0.532579615E-02
    0.228000000E+03    0.270607210E-01
    0.647100000E+02    0.101656846E+00
    0.210600000E+02    0.274574824E+00
    0.749500000E+01    0.448294319E+00
    0.279700000E+01    0.284902611E+00
    0.521500000E+00    0.151948592E-01
C     S
    0.666500000E+04   -0.293269653E-03
    0.100000000E+04   -0.231803547E-02
    0.228000000E+03   -0.114997860E-01
    0.647100000E+02   -0.468267270E-01
    0.210600000E+02   -0.128466169E+00
    0.749500000E+01   -0.301266272E+00
    0.279700000E+01   -0.255630702E+00
    0.521500000E+00    0.109379336E+01
C     S
    0.159600000E+00    0.100000000E+01
C     P
    0.943900000E+01    0.569792516E-01
    0.200200000E+01    0.313207212E+00
    0.545600000E+00    0.760376742E+00
C     P
    0.151700000E+00    0.100000000E+01
C     D
    0.550000000E+00    0.100000000E+01
''')}

# Remember to check the charge and spin
mol.charge = 0
mol.spin = 0
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
mf.max_cycle = 1
mf.max_memory = 1000 # MB
mf.kernel()

# read MOs from .fch(k) file
nbf = mf.mo_coeff.shape[0]
nif = mf.mo_coeff.shape[1]
mf.mo_coeff = fch2py('c2_uhf_gvb4_2CASSCF_NO.fch', nbf, nif, 'a')
# read done

# check if input MOs are orthonormal
S = mol.intor_symmetric('int1e_ovlp')
check_orthonormal(nbf, nif, mf.mo_coeff, S)

#dm = mf.make_rdm1()
#mf.max_cycle = 10
#mf.kernel(dm)

mc = mcscf.CASSCF(mf,6,(3,3)).state_average_([0.25,0.25,0.25,0.25])
mc.fcisolver.max_memory = 500 # MB
mc.max_memory = 500 # MB
mc.max_cycle = 200
#mc.fcisolver.spin = 0
mc.fcisolver.max_cycle = 100
mc.natorb = True
mc.verbose = 5
mc.kernel()

# save NOs into .fch file
#copyfile('c2_uhf_gvb4_2CASSCF_NO.fch')
#py2fch('c2_uhf_gvb4_2CASSCF_NO.fch',nbf,nif,mc.mo_coeff,'a',mc.mo_occ,True)
