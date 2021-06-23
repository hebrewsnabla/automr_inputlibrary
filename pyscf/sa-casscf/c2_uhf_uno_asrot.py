from pyscf import gto, scf
from fch2py import fch2py
from ortho import check_orthonormal
from pyscf import lib
import numpy as np
import os
from py2fch import py2fch
from uno import uno
from construct_vir import construct_vir
from lo import pm
from assoc_rot import assoc_rot
from shutil import copyfile

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

mf = scf.UHF(mol)
mf.max_cycle = 1
mf.max_memory = 1000 # MB
mf.kernel()

# read MOs from .fch(k) file
nbf = mf.mo_coeff[0].shape[0]
nif = mf.mo_coeff[0].shape[1]
alpha_coeff = fch2py('c2_uhf.fch', nbf, nif, 'a')
beta_coeff = fch2py('c2_uhf.fch', nbf, nif, 'b')
mf.mo_coeff = (alpha_coeff, beta_coeff)
# read done

# check if input MOs are orthonormal
S = mol.intor_symmetric('int1e_ovlp')
check_orthonormal(nbf, nif, mf.mo_coeff[0], S)
check_orthonormal(nbf, nif, mf.mo_coeff[1], S)

dm = mf.make_rdm1()
mf.max_cycle = 10
mf.kernel(dm)

# transform UHF canonical orbitals to UNO
na = np.sum(mf.mo_occ[0]==1)
nb = np.sum(mf.mo_occ[1]==1)
idx, noon, alpha_coeff = uno(nbf,nif,na,nb,mf.mo_coeff[0],mf.mo_coeff[1],S, 0.99999E+00)
alpha_coeff = construct_vir(nbf, nif, idx[1], alpha_coeff, S)
mf.mo_coeff = (alpha_coeff, beta_coeff)
# done transform

# save the UNO into .fch file
os.system('fch_u2r c2_uhf.fch')
os.rename('c2_uhf_r.fch', 'c2_uhf_uno.fch')
py2fch('c2_uhf_uno.fch',nbf,nif,mf.mo_coeff[0],'a',noon,True)
# save done

# associated rotation
npair = np.int64((idx[1]-idx[0]-idx[2])/2)
if(npair > 0):
  idx2 = idx[0] + npair - 1
  idx3 = idx2 + idx[2]
  idx1 = idx2 - npair
  idx4 = idx3 + npair
  occ_idx = range(idx1,idx2)
  vir_idx = range(idx3,idx4)
  print(idx1, idx2, idx3, idx4)
  occ_loc_orb = pm(mol.nbas,mol._bas[:,0],mol._bas[:,1],mol._bas[:,3],mol.cart,nbf,npair,mf.mo_coeff[0][:,occ_idx],S,'mulliken')
  vir_loc_orb = assoc_rot(nbf, npair, mf.mo_coeff[0][:,occ_idx], occ_loc_orb, mf.mo_coeff[0][:,vir_idx])
  mf.mo_coeff[0][:,occ_idx] = occ_loc_orb.copy()
  mf.mo_coeff[0][:,vir_idx] = vir_loc_orb.copy()
# localization done

# save associated rotation MOs into .fch(k) file
copyfile('c2_uhf_uno.fch', 'c2_uhf_uno_asrot.fch')
noon = np.zeros(nif)
py2fch('c2_uhf_uno_asrot.fch',nbf,nif,mf.mo_coeff[0],'a',noon,False)
