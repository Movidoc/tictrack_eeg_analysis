def analyse_merged_ttl_tics_imitated(merged_ttl_tics,
                                    phase_start_key='start_spont',
                                    phase_end_key='end_spont',
                                    max_t_after_end=1.0):
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
                if k.startswith('start_i'):
                    found_back = ('start_i', v)
                    break
                elif k.startswith('end_i'):
                    found_back = ('end_i', v)
                    break
                elif k =='D':
                    found_back = ('D', v)
                    break

            # Case 2 : T after start_i
            if found_back is not None and found_back[0] == 'start_i':
                results_list.append({
                    "type": "start_then_T",
                    "start_time": found_back[1]  
                })
                print(f"start_tic at {found_back[1]} then T pressed at {value}")

            # Case 3 : T after end_i 
            elif found_back is not None and found_back[0] == 'end_i':
                time_after_end = value - found_back[1]
                if time_after_end <= max_t_after_end:
                    # find the start of the tic before end_i, backward search
                    start_found = None
                    for j in range(i-1, -1, -1):
                        key_k, value_k = next(iter(phase_events[j].items()))
                        if key_k.startswith('start_i'):
                            start_found = value_k
                            break
                    results_list.append({
                        "type": "end_then_T",
                        "start_time": start_found,
                    })
                    print(f"end_tic at {found_back[1]} then T pressed at {value}")
            
            # Case 4 : T before start_i
                else:
                    found_forward = None
                    for j_forward in range(i+1, len(phase_events)):
                        k_fwd, v_fwd = next(iter(phase_events[j_forward].items()))
                        if k_fwd.startswith('start_i'):
                            found_forward = ('start_i', v_fwd)
                            break
                    results_list.append({
                        "type": "T_before_start",
                        "T_time": value,
                        })
                    print(f"T pressed at {value} then tic starts at {found_forward[1]}")

            # Case 5 : Real tic during imitated tic 
            # elif found_back is not None and found_back[0] == 'D':
            #     results_list.append({
            #         "type": "D_then_T",
            #         "D_time": found_back[1],
            #         })
            #     print(f"T pressed at {value} during imitated tic started at {found_back[1]}")

        # Imitated tic 
        if key == 'D':

            found_back = None
            for j in range(i-1, -1, -1):
                k, v = next(iter(phase_events[j].items()))
                if k.startswith('start_i'):
                    found_back = ('start_i', v)
                    break
                elif k.startswith('end_i'):
                    found_back = ('end_i', v)
                    break
                elif k =='D':
                    found_back = ('D', v)
                    break

            # Case 1: D after start_i
            if found_back is not None and found_back[0] == 'start_i':
                results_list.append({
                    "type": "start_then_D",
                    "start_time": found_back[1],
                })
                print(f"start_tic at {found_back[1]} then D pressed at {value}")
            
            # Case 2 : D after D 
            elif found_back is not None and found_back[0] == 'D':
                results_list.append({
                    "type": "D_then_D",
                    "D_time": value,
                })
                print(f"D pressed at {value} then D pressed at {found_back[1]}")

            # Case 3 : D after end_i
            else:
                found_forward = None
                for j_forward in range(i+1, len(phase_events)):
                    k_fwd, v_fwd = next(iter(phase_events[j_forward].items()))
                    if k_fwd.startswith('start_i'):
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
