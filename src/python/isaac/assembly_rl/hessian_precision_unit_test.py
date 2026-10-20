"""CPU checks of the exact kernels used by the guarded H64 overlay."""

import subprocess
import sys

import pytest

from isaac.assembly_rl import hessian_precision


def test_source_hash_gate():
    with pytest.raises(RuntimeError, match="exact audited runtime"):
        hessian_precision.transform_sources(b"modified", b"modified")


@pytest.mark.parametrize(
    "body_count,cone", [(4, "pyramidal"), (6, "pyramidal"), (6, "elliptic")]
)
def test_cpu_sparse_assembly_factor_reuse_and_guards(body_count, cone):
    code = r"""
import pathlib, sys
import mujoco, mujoco_warp, numpy as np, warp as wp
from mujoco_warp._src import solver, block_cholesky, types, forward
from isaac.assembly_rl import hessian_precision as h
original_sources = [pathlib.Path(m.__file__).read_bytes() for m in (solver, block_cholesky)]
original_kernel = solver._update_gradient_cholesky_blocked(16, 48, 36)
original_small = solver._update_gradient_cholesky(24, False)
original_solve = solver.solve
h.install()
ov = sys.modules[h.SOLVER_MODULE]
assert forward.solver.solve is ov.solve
assert ov._update_gradient_cholesky_blocked_assembly_h64_v3(16, 48, 36) is not original_kernel
assert h.install() == h.fingerprint()
assert ov._update_gradient_cholesky_assembly_h64_v3(24, False) is not original_small
xml = '<mujoco><option jacobian="sparse" integrator="implicitfast" cone="CONE_NAME"/><worldbody><geom type="plane" size="10 10 .1"/>'
xml += ''.join(f'<body pos="{i} 0 .099"><freejoint/><geom type="box" size=".1 .1 .1" mass="1" condim="6" friction=".8 .05 .02"/></body>' for i in range(BODY_COUNT))
xml += '</worldbody></mujoco>'
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m,d)
with wp.ScopedDevice('cpu'):
    mw = mujoco_warp.put_model(m)
    dw = mujoco_warp.put_data(m,d,nworld=1,nconmax=32,njmax=256)
    ctx = ov._create_solver_context(mw,dw)
    assert ctx.h.dtype is wp.float64
    assert ctx.hfactor.size == 0 or ctx.hfactor.dtype is wp.float64
    # Exercise actual sparse assembly and blocked solve with native contact rows.
    ov.init_context(mw,dw,ctx)
    nefc = int(dw.nefc.numpy()[0])
    assert nefc > 0
    J = np.zeros((nefc,mw.nv),np.float64)
    values = dw.efc.J.numpy()[0,0]
    cols = dw.efc.J_colind.numpy()[0,0]
    for row,(start,count) in enumerate(zip(dw.efc.J_rowadr.numpy()[0,:nefc],dw.efc.J_rownnz.numpy()[0,:nefc])):
        J[row,cols[start:start+count]] = values[start:start+count]
    elem = mw.M_elemid.numpy()
    M = np.zeros((mw.nv,mw.nv),np.float64)
    for i in range(mw.nv):
        for j in range(mw.nv):
            index = elem[max(i,j),min(i,j)]
            if index >= 0: M[i,j] = float(dw.M.numpy()[0,index])
    weights = dw.efc.D.numpy()[0,:nefc].astype(np.float64)
    weights *= dw.efc.state.numpy()[0,:nefc] == types.ConstraintState.QUADRATIC.value
    expected = M + J.T @ (weights[:,None]*J)
    upper = np.triu(ctx.h.numpy()[0,:mw.nv,:mw.nv])
    actual = upper + np.triu(upper,1).T
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-12)
    assert np.linalg.eigvalsh(actual).min() > 0
    if mw.opt.cone == types.ConeType.ELLIPTIC:
        # Force actual native six-dimensional contact rows into the middle
        # cone zone, then compare J.T C J to an independent analytic Hessian.
        jar = ctx.Jaref.numpy()
        contacts = int(dw.nacon.numpy()[0])
        addr = dw.contact.efc_address.numpy()[:,0]
        dims = dw.contact.dim.numpy()
        friction = dw.contact.friction.numpy().astype(np.float64)
        for contact in range(contacts):
            head = int(addr[contact]); dim = int(dims[contact])
            assert dim == 6
            jar[0,head:head+dim] = [-.3,2.1,-1.2,.8,-.6,.9]
        ctx.Jaref.assign(jar);ctx.done.zero_()
        ov._update_constraint(mw,dw,ctx)
        assert np.count_nonzero(dw.efc.state.numpy()[0,:nefc] == types.ConstraintState.CONE) == nefc
        ov._update_gradient(mw,dw,ctx)
        expected = M.copy()
        for contact in range(contacts):
            head = int(addr[contact]); dim = int(dims[contact])
            mu = friction[contact,0] * float(mw.opt.impratio_invsqrt.numpy()[0])
            scales = np.r_[mu,friction[contact,:dim-1]]
            u = jar[0,head:head+dim].astype(np.float64)*scales
            tangent = np.linalg.norm(u[1:]); direction=u[1:]/tangent
            dm = float(dw.efc.D.numpy()[0,head])/(mu*mu*(1+mu*mu))
            C = np.zeros((dim,dim),np.float64)
            C[0,0] = 1
            C[0,1:] = -mu*direction;C[1:,0] = C[0,1:]
            C[1:,1:] = (mu*mu-mu*u[0]/tangent)*np.eye(dim-1)+(mu*u[0]/tangent)*np.outer(direction,direction)
            C *= dm * np.outer(scales,scales)
            expected += J[head:head+dim].T @ C @ J[head:head+dim]
        upper = np.triu(ctx.h.numpy()[0,:mw.nv,:mw.nv])
        actual = upper + np.triu(upper,1).T
        np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=3e-12)
        assert not ov._use_incremental(mw)
    gradient = ctx.grad.numpy()[0,:mw.nv].astype(np.float64)
    np.testing.assert_allclose(ctx.search.numpy()[0],-np.linalg.solve(expected,gradient),rtol=3e-6,atol=2e-6)
    # Reuse the existing double factor with a different float32 gradient.
    ctx.grad.assign(np.ones((1,mw.nv_pad),np.float32))
    ctx.done.zero_(); ctx.quad_changed_count.zero_(); ctx.state_changed_count.fill_(1)
    ov._cholesky_factorize_solve(mw,dw,ctx,skip_unchanged=mw.opt.cone == types.ConeType.PYRAMIDAL)
    np.testing.assert_allclose(ctx.search.numpy()[0],-np.linalg.solve(expected,np.ones(mw.nv)),rtol=3e-6,atol=2e-6)
    # Reproduce float32 loss of positive definiteness without changing equations.
    rng = np.random.default_rng(0)
    q,_ = np.linalg.qr(rng.normal(size=(mw.nv,mw.nv)))
    mass = (q*np.linspace(3e-4,1,mw.nv)) @ q.T
    jacobian = rng.normal(size=(12,mw.nv)).astype(np.float32)
    diagonal = np.geomspace(2e5,2e7,12).astype(np.float32)
    exact = mass.astype(np.float32).astype(np.float64)
    rounded = mass.astype(np.float32)
    for row,weight in zip(jacobian,diagonal):
        exact += float(weight)*np.outer(row.astype(np.float64),row.astype(np.float64))
        rounded += weight*np.outer(row,row)
    assert np.linalg.eigvalsh(rounded.astype(np.float64)).min() < 0
    assert np.linalg.eigvalsh(exact).min() > 0
    padded = np.eye(mw.nv_pad,dtype=np.float64)
    padded[:mw.nv,:mw.nv] = exact
    ctx.h.assign(padded[None]); ctx.done.zero_()
    ov._cholesky_factorize_solve(mw,dw,ctx)
    np.testing.assert_allclose(ctx.search.numpy()[0],-np.linalg.solve(exact,np.ones(mw.nv)),rtol=3e-5,atol=3e-6)
    # The full native iteration exercises double sparse incremental updates too.
    ov.solve(mw,dw)
    assert np.isfinite(dw.qacc.numpy()).all()
    for owner, attr, bad in ((mw,'is_sparse',False),(mw.opt,'integrator',types.IntegratorType.EULER),(mw.opt,'enableflags',int(types.EnableBit.SLEEP)),(mw.opt,'solver',types.SolverType.CG)):
        old = getattr(owner,attr);setattr(owner,attr,bad)
        try:
            try: ov.solve(mw,dw)
            except ValueError as error:
                assert 'H64 requires' in str(error)
                assert all(key+'=' in str(error) for key in ('is_sparse','solver','cone','integrator','enableflags','nv'))
            else: raise AssertionError('unsupported solver was accepted')
        finally: setattr(owner,attr,old)
assert [pathlib.Path(m.__file__).read_bytes() for m in (solver,block_cholesky)] == original_sources
solver.solve = original_solve
try: h.install()
except RuntimeError as error: assert 'binding conflicts' in str(error)
else: raise AssertionError('conflicting binding accepted')
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code.replace("BODY_COUNT", str(body_count)).replace("CONE_NAME", cone),
        ],
        check=True,
        timeout=120,
    )
