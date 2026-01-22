def analyse_merged_ttl_tics_imitated(merged_ttl_tics,
                                    phase_start_key='start_spont',
                                    phase_end_key='end_spont'):
    """
    We devide tics into real and imitated. For real tics we check if the T occured. For imitated we check D and F keys if they occured. 
    For T occuring without any tic around it (from excel) we disregard the T (treating it as an error)---- might add later 
    """

    t_start = next((list(d.values())[0] for d in merged_ttl_tics
                    if list(d.keys())[0] == phase_start_key), None)
    t_end   = next((list(d.values())[0] for d in merged_ttl_tics
                    if list(d.keys())[0] == phase_end_key), None)

    phase_events = [d for d in merged_ttl_tics
                    if t_start <= list(d.values())[0] <= t_end]

    results_list = []
    i = 0

    while i < len(phase_events):

        key, value = next(iter(phase_events[i].items()))

        # Real tic 
        if key == 'T':

            found_back = None
            for j in range(i-1, -1, -1):
                k, v = next(iter(phase_events[j].items()))
                if k.startswith('start_'):
                    found_back = ('start_i', v)
                    break
                elif k.startswith('end_'):
                    found_back = ('end_i', v)
                    break

            # Scan forward for next start_i if needed
            found_forward = None
            for j_forward in range(i+1, len(phase_events)):
                k_fwd, v_fwd = next(iter(phase_events[j_forward].items()))
                if k_fwd.startswith('start_'):
                    found_forward = v_fwd
                    break

            # Case 1 : Discard isolated T (no start_i before or after)
            if found_back is None and found_forward is None:
                i += 1
                continue

            # Case 2 : T after start_i
            if found_back is not None and found_back[0] == 'start_i':
                results_list.append({
                    "type": "T_during_tic",
                    "start_time": found_back[1]  
                })
                print(f"start_tic at {found_back[1]} then T pressed at {value}")
            else:
                anchor_time = value  # T before or after tic
                results_list.append({
                    "type": "T_outside_tic",
                    "anchor_time": anchor_time
                })
                if found_forward is not None:
                    print(f"T pressed at {value} then tic starts at {found_forward}")
                else:
                    print(f"T pressed at {value} after tic ended")

        # Imitated tic 
        if key == 'D':

            found_back = None
            for j in range(i-1, -1, -1):
                k, v = next(iter(phase_events[j].items()))
                if k.startswith('start_'):
                    found_back = ('start_i', v)
                    break
                elif k.startswith('end_'):
                    found_back = ('end_i', v)
                    break

            # Case 1: D after start_i
            if found_back is not None and found_back[0] == 'start_i':
                results_list.append({
                    "type": "start_then_D",
                    "start_time": found_back[1],
                })
                print(f"start_tic at {found_back[1]} then D pressed at {value}")

            # Case 2: D after end_i
            else:
                found_forward = None
                for j_forward in range(i+1, len(phase_events)):
                    k_fwd, v_fwd = next(iter(phase_events[j_forward].items()))
                    if k_fwd.startswith('start_'):
                        found_forward = ('start_i', v_fwd)
                        break
                    elif k_fwd == 'F':
                        found_forward = ('F', v_fwd)
                        break

                if found_forward is not None and found_forward[0] == 'start_i':
                    results_list.append({
                        "type": "D_then_start",
                        "D_time": value,
                        })
                    print(f"D pressed at {value} then tic starts at {found_forward[1]}")
                else:
                    results_list.append({
                        "type": "D_then_F",
                        "D_time": value
                    })
                    print(f"D pressed at {value} without any visible tic then F pressed at {found_back[1]}")

        i += 1

    return results_list
