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

lib.num_threads(8)

mol = gto.M()
# 2 atom(s)
mol.atom = '''
C1              0.00000000        0.00000000        0.00000000
C2              0.00000000        0.00000000        2.00000001
'''

mol.basis = {
'C1': gto.basis.parse('''
C     S
    0.135753497E+05    0.606455036E-03
    0.203523337E+04    0.469790889E-02
    0.463225624E+03    0.243324776E-01
    0.131200196E+03    0.973999683E-01
    0.428530159E+02    0.301955843E+00
    0.155841858E+02    0.662336091E+00
C     S
    0.620671385E+01    0.655953085E+00
    0.257648965E+01    0.375856333E+00
C     S
    0.576963394E+00    0.100000000E+01
C     S
    0.229728314E+00    0.100000000E+01
C     S
    0.951644400E-01    0.100000000E+01
C     P
    0.346972322E+02    0.113272200E-01
    0.795826228E+01    0.761696592E-01
    0.237808269E+01    0.301922522E+00
    0.814332082E+00    0.727850273E+00
C     P
    0.288875473E+00    0.100000000E+01
C     P
    0.100568237E+00    0.100000000E+01
C     D
    0.109700000E+01    0.100000000E+01
C     D
    0.318000000E+00    0.100000000E+01
C     F
    0.761000000E+00    0.100000000E+01
'''),
'C2': gto.basis.parse('''
C     S
    0.135753497E+05    0.606455036E-03
    0.203523337E+04    0.469790889E-02
    0.463225624E+03    0.243324776E-01
    0.131200196E+03    0.973999683E-01
    0.428530159E+02    0.301955843E+00
    0.155841858E+02    0.662336091E+00
C     S
    0.620671385E+01    0.655953085E+00
    0.257648965E+01    0.375856333E+00
C     S
    0.576963394E+00    0.100000000E+01
C     S
    0.229728314E+00    0.100000000E+01
C     S
    0.951644400E-01    0.100000000E+01
C     P
    0.346972322E+02    0.113272200E-01
    0.795826228E+01    0.761696592E-01
    0.237808269E+01    0.301922522E+00
    0.814332082E+00    0.727850273E+00
C     P
    0.288875473E+00    0.100000000E+01
C     P
    0.100568237E+00    0.100000000E+01
C     D
    0.109700000E+01    0.100000000E+01
C     D
    0.318000000E+00    0.100000000E+01
C     F
    0.761000000E+00    0.100000000E+01
''')}

# Remember to check the charge and spin
mol.charge = 0
mol.spin = 0
mol.verbose = 4
mol.build()

mf = scf.UHF(mol)
mf.max_cycle = 1
mf.max_memory = 10000 # MB
mf.kernel()

# read MOs from .fch(k) file
nbf = mf.mo_coeff[0].shape[0]
nif = mf.mo_coeff[0].shape[1]
alpha_coeff = fch2py('C2_20_uhf.fch', nbf, nif, 'a')
beta_coeff = fch2py('C2_20_uhf.fch', nbf, nif, 'b')
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
os.system('fch_u2r C2_20_uhf.fch')
os.rename('C2_20_uhf_r.fch', 'C2_20_uhf_uno.fch')
py2fch('C2_20_uhf_uno.fch',nbf,nif,mf.mo_coeff[0],'a',noon,True)
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
copyfile('C2_20_uhf_uno.fch', 'C2_20_uhf_uno_asrot.fch')
noon = np.zeros(nif)
py2fch('C2_20_uhf_uno_asrot.fch',nbf,nif,mf.mo_coeff[0],'a',noon,False)
