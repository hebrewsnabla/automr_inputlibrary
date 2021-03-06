from pyscf import gto, scf
from fch2py import fch2py
from ortho import check_orthonormal
from pyscf import mcscf, lib
from py2fch import py2fch
from shutil import copyfile
import numpy as np

lib.num_threads(8)

mol = gto.M()
# 2 atom(s)
mol.atom = '''
Cr1             0.00000000        0.00000000        0.00000000
Cr2             0.00000000        0.00000000        4.00000002
'''

mol.basis = {
'Cr1': gto.basis.parse('''
Cr    S
    0.254477807E+06    0.242904776E-03
    0.381317971E+05    0.188435236E-02
    0.867529306E+04    0.980095844E-02
    0.245500998E+04    0.398250087E-01
    0.799162178E+03    0.129405437E+00
    0.286900215E+03    0.306290018E+00
    0.111254132E+03    0.434628350E+00
    0.438641526E+02    0.224695629E+00
Cr    S
    0.279326692E+03   -0.243633579E-01
    0.862747324E+02   -0.115114954E+00
    0.135557561E+02    0.550922663E+00
    0.569781128E+01    0.536113547E+00
Cr    S
    0.856365826E+01   -0.371230177E+00
    0.139882968E+01    0.116811695E+01
Cr    S
    0.572881711E+00    0.100000000E+01
Cr    S
    0.900961713E-01    0.100000000E+01
Cr    S
    0.341258851E-01    0.100000000E+01
Cr    P
    0.130643989E+04    0.282928007E-02
    0.309253114E+03    0.227766292E-01
    0.989962740E+02    0.105645619E+00
    0.367569165E+02    0.299499449E+00
    0.145666571E+02    0.477062454E+00
    0.587399374E+01    0.276542344E+00
Cr    P
    0.228909997E+02   -0.194121061E-01
    0.308550018E+01    0.386188758E+00
    0.121323291E+01    0.676239089E+00
Cr    P
    0.449316807E+00    0.100000000E+01
Cr    D
    0.437200745E+02    0.220932593E-01
    0.123912427E+02    0.128014388E+00
    0.426394420E+01    0.386529103E+00
    0.155252218E+01    0.641033014E+00
Cr    D
    0.537619295E+00    0.100000000E+01
Cr    D
    0.164931731E+00    0.100000000E+01
Cr    P
    0.120675000E+00    0.100000000E+01
Cr    D
    0.510000000E-01    0.100000000E+01
Cr    F
    0.114700000E+01    0.100000000E+01
'''),
'Cr2': gto.basis.parse('''
Cr    S
    0.254477807E+06    0.242904776E-03
    0.381317971E+05    0.188435236E-02
    0.867529306E+04    0.980095844E-02
    0.245500998E+04    0.398250087E-01
    0.799162178E+03    0.129405437E+00
    0.286900215E+03    0.306290018E+00
    0.111254132E+03    0.434628350E+00
    0.438641526E+02    0.224695629E+00
Cr    S
    0.279326692E+03   -0.243633579E-01
    0.862747324E+02   -0.115114954E+00
    0.135557561E+02    0.550922663E+00
    0.569781128E+01    0.536113547E+00
Cr    S
    0.856365826E+01   -0.371230177E+00
    0.139882968E+01    0.116811695E+01
Cr    S
    0.572881711E+00    0.100000000E+01
Cr    S
    0.900961713E-01    0.100000000E+01
Cr    S
    0.341258851E-01    0.100000000E+01
Cr    P
    0.130643989E+04    0.282928007E-02
    0.309253114E+03    0.227766292E-01
    0.989962740E+02    0.105645619E+00
    0.367569165E+02    0.299499449E+00
    0.145666571E+02    0.477062454E+00
    0.587399374E+01    0.276542344E+00
Cr    P
    0.228909997E+02   -0.194121061E-01
    0.308550018E+01    0.386188758E+00
    0.121323291E+01    0.676239089E+00
Cr    P
    0.449316807E+00    0.100000000E+01
Cr    D
    0.437200745E+02    0.220932593E-01
    0.123912427E+02    0.128014388E+00
    0.426394420E+01    0.386529103E+00
    0.155252218E+01    0.641033014E+00
Cr    D
    0.537619295E+00    0.100000000E+01
Cr    D
    0.164931731E+00    0.100000000E+01
Cr    P
    0.120675000E+00    0.100000000E+01
Cr    D
    0.510000000E-01    0.100000000E+01
Cr    F
    0.114700000E+01    0.100000000E+01
''')}

# Remember to check the charge and spin
mol.charge = 0
mol.spin = 0
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
mf.max_cycle = 1
mf.max_memory = 10000 # MB
mf.kernel()

# read MOs from .fch(k) file
nbf = mf.mo_coeff.shape[0]
nif = mf.mo_coeff.shape[1]
mf.mo_coeff = fch2py('Cr2_40_uhf_SUHFNO.fch', nbf, nif, 'a')
# read done

# check if input MOs are orthonormal
S = mol.intor_symmetric('int1e_ovlp')
check_orthonormal(nbf, nif, mf.mo_coeff, S)

#dm = mf.make_rdm1()
#mf.max_cycle = 10
#mf.kernel(dm)

mc = mcscf.CASSCF(mf,12,(6,6))
mc.fcisolver.max_memory = 5000 # MB
mc.max_memory = 5000 # MB
mc.max_cycle = 200
mc.fcisolver.spin = 0
mc.fix_spin_(ss=0)
mc.fcisolver.level_shift = 0.2
mc.fcisolver.pspace_size = 1200
mc.fcisolver.max_space = 100
mc.fcisolver.max_cycle = 300
mc.natorb = True
mc.verbose = 5
mc.kernel()

# save NOs into .fch file
copyfile('Cr2_40_uhf_SUHFNO.fch', 'Cr2_40_uhf_SUHFNO_CASSCF_NO.fch')
py2fch('Cr2_40_uhf_SUHFNO_CASSCF_NO.fch',nbf,nif,mc.mo_coeff,'a',mc.mo_occ,True)
