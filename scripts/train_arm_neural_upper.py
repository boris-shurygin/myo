"""Warm-start the upper policy, then PPO on real reference position/velocity reward."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from myo.arm_hierarchy import Motion, VALIDATION_MOTIONS, TEST_MOTIONS, evaluate, load_policy
from myo.arm_learning import ArmPPO
from myo.arm_neural_upper import NeuralUpperEnv, save_upper, load_upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lower', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--warm-updates', type=int, default=2000)
    parser.add_argument('--ppo-steps', type=int, default=32768)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    lower, physical, _ = load_policy(args.lower)
    env = NeuralUpperEnv(lower.policy, physical.contract())
    physical.close()
    frozen_lower, lower_contract = lower.policy, env.lower_contract
    factories = [lambda i=i: NeuralUpperEnv(frozen_lower,lower_contract,worker=i,workers=4) for i in range(4)]
    vec = SubprocVecEnv(factories,start_method='forkserver')
    validation = VALIDATION_MOTIONS+(Motion('val_lower133',133.,122.,rise_s=1.5,kind='reach'),
                                    Motion('val_hold131',131.,131.,kind='hold'))
    model = ArmPPO('MlpPolicy', vec, seed=0,device='cpu',learning_rate=1e-5,
        n_steps=256,batch_size=128,n_epochs=10,gamma=.99,gae_lambda=.95,clip_range=.2,
        ent_coef=0.,vf_coef=.5,max_grad_norm=.5,target_kl=.03,
        policy_kwargs={'net_arch':{'pi':[128]*4,'vf':[128]*4},'activation_fn':nn.SiLU,
            'log_std_init':-3.,'optimizer_class':torch.optim.RMSprop,'optimizer_kwargs':{'eps':1e-5}})
    model.set_logger(configure(str(args.out),['csv']))
    training = {'lower_sha256':hashlib.sha256(args.lower.read_bytes()).hexdigest(),
        'architecture':'private actor/critic 4x128 SiLU; Gaussian scalar acceleration, clipped to[-1,1]',
        'upper_input_count':len(env.observation_names),'warm_optimizer':'Adam1e-3',
        'warm_data':'teacher rollouts plus independent reference-error jitter: q0.1rad,v0.2rad/s,a0.5rad/s2; physical sensors unchanged',
        'ppo_optimizer':'RMSprop1e-5','ppo_reward':'reference q/v, no heuristic action target, no effort',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    states, labels = [], []
    try:
        for motion in env.env.motions:
            observation,_ = env.reset(seed=0,options={'motion':motion})
            while True:
                teacher = env.heuristic_action()
                if env.env.step_index < round(motion.seconds/.02):
                    qerr,verr,aff,acc = observation[-4:]
                    expected = np.clip(5*aff+16*(qerr-.05*4*observation[1]-.5*.05**2*5*acc)
                                       +8*(verr-.05*5*acc),-20,20)/20
                    if not np.isclose(expected,teacher[0],atol=2e-6):
                        raise AssertionError('Upper feature/teacher timing mismatch')
                    states.append(observation.copy()); labels.append(teacher)
                observation,_,failed,done,_ = env.step(teacher)
                if failed or done:
                    break
    except BaseException:
        vec.close(); env.close()
        raise
    x,y = torch.tensor(np.array(states)),torch.tensor(np.array(labels))
    parameters = [p for name,p in model.policy.named_parameters()
                  if name.startswith('mlp_extractor.policy_net.') or name.startswith('action_net.')]
    optimizer = torch.optim.Adam(parameters,lr=1e-3)
    rng = torch.Generator().manual_seed(0)
    history, best, best_phase = [], None, None
    def check(phase,step):
        nonlocal best,best_phase
        report = evaluate(model,env,validation)
        rank = (report['failures'],-report['passed'],report['mean_position_rmse_deg'])
        row = {'phase':phase,'step':step,'validation':report,'audit':model.last_audit}
        history.append(row)
        if best is None or rank < best:
            best,best_phase = rank,phase
            save_upper(model,env,args.out/'best.zip',training|{'selected_phase':phase,'selected_step':step})
        (args.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps({'phase':phase,'step':step,'passed':report['passed'],
                          'q_rmse_deg':report['mean_position_rmse_deg']}),flush=True)
    try:
        for update in range(1,args.warm_updates+1):
            index = torch.randint(len(x),(256,),generator=rng)
            batch = x[index].clone()
            batch[:,-4:-1] += torch.randn((256,3),generator=rng)*torch.tensor([.1,.2,.1])
            qerr,verr,aff,acc = batch[:,-4:].unbind(-1)
            target = (5*aff+16*(qerr-.05*4*batch[:,1]-.5*.05**2*5*acc)
                      +8*(verr-.05*5*acc)).clamp(-20,20)/20
            action = model.policy.get_distribution(batch).get_actions(deterministic=True)
            loss = (action[:,0]-target).square().mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.); optimizer.step()
            if update % 500 == 0 or update == args.warm_updates:
                check('imitation',update)
        selected, replay, selected_metadata = load_upper(args.out/'best.zip')
        replay.close()
        model.policy.load_state_dict(selected.policy.state_dict())
        try:
            for _ in range(0,args.ppo_steps,16384):
                model.learn(min(16384,args.ppo_steps-model.num_timesteps),reset_num_timesteps=False)
                check('PPO',model.num_timesteps)
        finally:
            vec.close()
        save_upper(model,env,args.out/'last.zip',training|{'selected_phase':'last_PPO','selected_step':model.num_timesteps})
        selected,replay,_ = load_upper(args.out/'best.zip')
        try:
            result = {name:evaluate(selected,replay,motions,traces=True) for name,motions in
                [('train',env.env.motions),('validation',validation),('test',TEST_MOTIONS)]}
            (args.out/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            print(json.dumps({'selected_phase':best_phase,**{name:{k:v for k,v in r.items() if k!='cases'}
                  for name,r in result.items()}}),flush=True)
        finally:
            replay.close()
    finally:
        vec.close()
        env.close()


if __name__ == '__main__':
    main()
