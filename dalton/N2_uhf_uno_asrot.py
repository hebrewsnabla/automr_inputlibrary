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
N1              0.00000000        0.00000000        0.00000000
N2              0.00000000        0.00000000        2.00000000
'''

mol.basis = {
'N1': gto.basis.parse('''
N     S
    0.904600000E+04    0.699617413E-03
    0.135700000E+04    0.538605463E-02
    0.309300000E+03    0.273910212E-01
    0.877300000E+02    0.103150592E+00
    0.285600000E+02    0.278570663E+00
    0.102100000E+02    0.448294849E+00
    0.383800000E+01    0.278085928E+00
    0.746600000E+00    0.154315612E-01
N     S
    0.904600000E+04   -0.304990096E-03
    0.135700000E+04   -0.240802638E-02
    0.309300000E+03   -0.119444487E-01
    0.877300000E+02   -0.489259929E-01
    0.285600000E+02   -0.134472725E+00
    0.102100000E+02   -0.315112578E+00
    0.383800000E+01   -0.242857833E+00
    0.746600000E+00    0.109438221E+01
N     S
    0.224800000E+00    0.100000000E+01
N     P
    0.135500000E+02    0.589056768E-01
    0.291700000E+01    0.320461107E+00
    0.797300000E+00    0.753042062E+00
N     P
    0.218500000E+00    0.100000000E+01
N     D
    0.817000000E+00    0.100000000E+01
'''),
'N2': gto.basis.parse('''
N     S
    0.904600000E+04    0.699617413E-03
    0.135700000E+04    0.538605463E-02
    0.309300000E+03    0.273910212E-01
    0.877300000E+02    0.103150592E+00
    0.285600000E+02    0.278570663E+00
    0.102100000E+02    0.448294849E+00
    0.383800000E+01    0.278085928E+00
    0.746600000E+00    0.154315612E-01
N     S
    0.904600000E+04   -0.304990096E-03
    0.135700000E+04   -0.240802638E-02
    0.309300000E+03   -0.119444487E-01
    0.877300000E+02   -0.489259929E-01
    0.285600000E+02   -0.134472725E+00
    0.102100000E+02   -0.315112578E+00
    0.383800000E+01   -0.242857833E+00
    0.746600000E+00    0.109438221E+01
N     S
    0.224800000E+00    0.100000000E+01
N     P
    0.135500000E+02    0.589056768E-01
    0.291700000E+01    0.320461107E+00
    0.797300000E+00    0.753042062E+00
N     P
    0.218500000E+00    0.100000000E+01
N     D
    0.817000000E+00    0.100000000E+01
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
alpha_coeff = fch2py('N2_uhf.fch', nbf, nif, 'a')
beta_coeff = fch2py('N2_uhf.fch', nbf, nif, 'b')
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
os.system('fch_u2r N2_uhf.fch')
os.rename('N2_uhf_r.fch', 'N2_uhf_uno.fch')
py2fch('N2_uhf_uno.fch',nbf,nif,mf.mo_coeff[0],'a',noon,True)
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
copyfile('N2_uhf_uno.fch', 'N2_uhf_uno_asrot.fch')
noon = np.zeros(nif)
py2fch('N2_uhf_uno_asrot.fch',nbf,nif,mf.mo_coeff[0],'a',noon,False)
