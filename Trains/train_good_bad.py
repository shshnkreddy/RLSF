import sys
sys.path.append('..')
sys.path.append('./')
from Parameters.SIM_parameters import *
import pickle

#------------------------------------------#
def main():
    import wandb
    from Sources.utils import create_folder
    name = f'{mode}-{env_name}'
    if(wandb_log):
        wandb.init(name=name, project='PREF')
        wandb.config.update(args)
        
    from Sources.utils import create_folder
    
    global weight_path
    weight_path = f'./weights/{env_name}/{mode}'
    create_folder(weight_path)

    #------------------------------------------#
    import safety_gymnasium
    from Sources.wrapper import CostWrapper

    if('Safety' in env_name):
        sample_env = safety_gymnasium.make(env_name)
        wrappers = [CostWrapper]

        env = safety_gymnasium.vector.make(env_id=env_name, num_envs=num_envs, wrappers=wrappers)
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.make(env_id=env_name, num_envs=eval_num_envs, wrappers=wrappers)
        else:
            test_env = None

    elif('Driver' in env_name):
        from Sources.envs import get_driver
        if('blocking' in env_name.lower()):
            scenario = 'blocked'
        elif('two' in env_name.lower()):
            scenario = 'twolanes'
        elif('change' in env_name.lower()):
            scenario = 'changing_lane'

        sample_env = get_driver(scenario=scenario)
        env = safety_gymnasium.vector.SafetyAsyncVectorEnv([lambda: get_driver(scenario=scenario, constraint=True) for _ in range(num_envs)])
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.SafetyAsyncVectorEnv([lambda: get_driver(scenario=scenario, constraint=True) for _ in range(eval_num_envs)])
        else:
            test_env = None

    elif('BiasedPendulum' in env_name):
        import gymnasium as gym
        from Sources.wrapper import BiasedPendulumWrapper
        
        def BiasedPendulum():
            env = gym.make('InvertedPendulum-v4')
            env = BiasedPendulumWrapper(env)
            return env

        sample_env = BiasedPendulum()
        env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BiasedPendulum for _ in range(num_envs)])
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BiasedPendulum for _ in range(eval_num_envs)])
        else:
            test_env = None

    elif('BlockedSwimmer' in env_name):
        import gymnasium as gym
        from Sources.wrapper import BlockedSwimmerWrapper

        def BlockedSwimmer():
            env = gym.make('Swimmer-v4')
            env = BlockedSwimmerWrapper(env)
            return env
        
        sample_env = BlockedSwimmer()
        env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BlockedSwimmer for _ in range(num_envs)])
        if (eval_num_envs):
            test_env = safety_gymnasium.vector.SafetyAsyncVectorEnv([BlockedSwimmer for _ in range(eval_num_envs)])
        else:
            test_env = None

    else:
        raise ValueError('Unknown environment')
    
    #------------------------------------------#
    from Sources.algo.sim import SIM
    from Sources.buffer import Trajectory_Buffer_Continuous
    from Sources.network import StateIndependentPolicy
    from Sources.utils import create_folder
    import threading
    import torch
    import setproctitle
    from torch import nn

    def evaluate(algo, env,max_episode_length):
        global max_eval_return
        mean_return = 0.0
        mean_cost = 0.0
        failed_case = []
        cost_sum = [0 for _ in range(eval_num_envs)]

        for step in range(num_eval_episodes//eval_num_envs):
            state,_ = env.reset()
            episode_return = 0.0
            episode_cost = 0.0
            for iter in range(max_episode_length):
                if (iter%100 == 0):
                    print(f'valid {step+1}/{num_eval_episodes//eval_num_envs}: {iter/max_episode_length*100:.2f}% {iter}/{max_episode_length}', end='\r')
                action = algo.exploit(state)
                state, reward, cost, done, _, _ = env.step(action)
                episode_return += np.sum(reward)
                episode_cost += np.sum(cost)
                for idx in range(eval_num_envs):
                    cost_sum[idx] += cost[idx]
            for idx in range(eval_num_envs):
                failed_case.append(cost_sum[idx])
                cost_sum[idx] = 0
            mean_return += episode_return 
            mean_cost += episode_cost 

        mean_return = mean_return/num_eval_episodes
        mean_cost = mean_cost/num_eval_episodes
        tmp_arr = np.asarray(failed_case)

        success_rate = np.sum(tmp_arr<=cost_limit)/num_eval_episodes
        value = (mean_return * success_rate)/10
        if (value>max_eval_return):
            max_eval_return = value
            algo.save_models(f'{weight_path}/({value:.3f})-({success_rate:.2f})-({mean_return:.2f})-({mean_cost:.2f})')
        else:
            max_eval_return*=0.999
        print(f'[Eval] R: {mean_return:.2f}, C: {mean_cost:.2f}, '+
            f'SR: {success_rate:.2f}, '
            f'V: {value:.2f}, maxV: {max_eval_return:.2f}')

    def collect_expert_demonstration(env,algo,total_interaction=int(1e6)):
        t = [0 for _ in range(num_envs)]
        state,_ = env.reset()

        num_expert_steps = total_interaction//num_envs
        current_valid_return = 0
        current_valid_cost = 0
        current_good_rate = 0
        num_total_traj = 0
        print('collect good trajectories')
        for step in range(1,num_expert_steps+1):
            state, t,current_valid_return,current_valid_cost,current_good_rate,num_total_traj = algo.expert_step(env, state, t)
            if (step %1000 == 0):
                print(f'{step/num_expert_steps*100:.2f}%, return = {current_valid_return:.3f}, costs = {current_valid_cost:.3f},'+
                f' good_rate = {current_good_rate:.2f}, len = {int(num_total_traj*current_good_rate)}, {step}/{num_expert_steps}', end='\r')

        #save buffers
        buffer_dir = f'{weight_path}/expert_buffer'
        create_folder(buffer_dir)
        pickle.dump(algo.exp_good_buffer, open(f'{buffer_dir}/good.pkl', 'wb'))
        pickle.dump(algo.exp_bad_buffer, open(f'{buffer_dir}/bad.pkl', 'wb'))

    def train(env,test_env,algo,eval_algo):
        t = [0 for _ in range(num_envs)]
        eval_thread = None
        state,_ = env.reset()

        print('start training')
        for step in range(1,num_training_step//num_envs+1):
            if (step%100==0):
                print(f'train: {step/(num_training_step//num_envs)*100:.2f}% {step}/{num_training_step//num_envs}', end='\r')
                if(wandb_log):
                    wandb.log({'train/step': step/(num_training_step//num_envs)*100})
            state, t = algo.step(env, state, t)
            if algo.is_update(step*num_envs):
                    eval_return.write(f'{np.mean(algo.return_reward)}\n')
                    eval_return.flush()
                    eval_cost.write(f'{np.mean(algo.return_cost)}\n')
                    eval_cost.flush()
                    algo.update()
                    
            if step % (eval_interval//num_envs) == 0:
                algo.save_models(f'{weight_path}/s{seed}-latest')
                if (test_env):
                    if eval_thread is not None:
                        eval_thread.join()
                    eval_algo.copyNetworksFrom(algo)
                    eval_algo.eval()
                    eval_thread = threading.Thread(target=evaluate, 
                    args=(eval_algo,test_env,max_episode_length))
                    eval_thread.start()
        if(eval_thread is not None):
            eval_thread.join()
        algo.save_models(f'{weight_path}/s{seed}-finish')


    state_shape=sample_env.observation_space.shape
    action_shape=sample_env.action_space.shape
    sample_env.close()

    exp_good_buffer = Trajectory_Buffer_Continuous(
        buffer_size=1000000,
        state_shape=state_shape,
        action_shape=action_shape,
        device=device,
    )

    exp_bad_buffer = Trajectory_Buffer_Continuous(
        buffer_size=2000000,
        state_shape=state_shape,
        action_shape=action_shape,
        device=device,
    )    

    expert_actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=hidden_units_actor,
            hidden_activation=nn.ReLU()
        ).to(device)
    

    if (dynamic_good):
        print('training SIM with dynamic good threshold')
    else:
        print('training SIM with fixed good threshold')
    print(f'threshold: minG: {min_good}, maxB: {max_bad}')

    setproctitle.setproctitle(f'{env_name}-SIM-{seed}')
    algo = SIM(env_name=env_name,expert_actor=expert_actor,exp_good_buffer=exp_good_buffer,exp_bad_buffer=exp_bad_buffer,
            state_shape=state_shape, action_shape=action_shape,
            device=device, seed=seed, gamma=gamma,cost_gamma=cost_gamma,buffer_size=buffer_size,
            mix=mix, hidden_units_actor=hidden_units_actor,
            hidden_units_critic=hidden_units_critic,units_clfs=hidden_units_clfs,
            lr_actor=lr_actor,lr_critic=lr_critic,lr_cost_critic=lr_cost_critic,lr_penalty=lr_penalty, epoch_ppo=epoch_ppo,
            epoch_clfs=epoch_clfs,batch_size=batch_size,lr_clfs=lr_clfs,clip_eps=clip_eps, lambd=lambd, coef_ent=coef_ent,
            max_grad_norm=max_grad_norm,reward_factor=reward_factor,max_episode_length=max_episode_length,
            cost_limit=cost_limit,risk_level=risk_level,num_envs=num_envs,
            dynamic_good=dynamic_good,min_good=min_good,max_bad=max_bad,
            conf_coef=conf_coef,tanh_conf=tanh_conf,start_bad=start_bad, wandb_log=wandb_log, mode=mode)
    
    if(mode=='SIMKC'):
        buffer_dir = f'{weight_path}/expert_buffer'
        if(os.path.exists(buffer_dir)):
            algo.exp_good_buffer = pickle.load(open(f'{buffer_dir}/good.pkl', 'rb'))
            algo.exp_bad_buffer = pickle.load(open(f'{buffer_dir}/bad.pkl', 'rb'))
            print('Loaded expert demos from disk.')

        else:
            assert expert_path is not None
            expert_actor.load_state_dict(torch.load(expert_path))
            collect_expert_demonstration(env=env,algo=algo)

    eval_algo = None
    train(env=env,test_env=test_env,algo=algo,eval_algo=eval_algo)

    env.close()
    if (test_env):
        test_env.close()

    wandb.finish()

if __name__ == '__main__':
    main()