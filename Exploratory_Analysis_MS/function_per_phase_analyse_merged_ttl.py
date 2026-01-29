def analyse_merged_ttl_tics_imitated(merged_ttl_tics,
                                    phase_start_key='start_spont',
                                    phase_end_key='end_spont',
                                    max_t_after_end=1.0,
                                    max_t_after_F=2.0):
    """
    We devide tics into real and imitated. For real tics we check if the T occured. For imitated we check D and F keys if they occured. 
    For T occuring without any tic around it (from excel) we disregard the T (treating it as an error)---- might add later 
    """

    t_start = next((list(d.values())[0] for d in merged_ttl_tics
                    if list(d.keys())[0] == phase_start_key), None)
    t_end   = next((list(d.values())[0] for d in merged_ttl_tics
                    if list(d.keys())[0] == phase_end_key), None)

    phase_events = [d for d in merged_ttl_tics
                    if t_start < list(d.values())[0] < t_end]

    results_list = []
    i = 0
    D_time_used = set()

    while i < len(phase_events):

        key, value = next(iter(phase_events[i].items()))

        # ---------- Real tic -----------
        if key == 'T':

            found_back = None
            for j in range(i-1, -1, -1):
                k, v = next(iter(phase_events[j].items()))
                if k.startswith('start_'):
                    found_back = ('start_', v)
                    break
                elif k.startswith('end_i'):
                    found_back = ('end_i', v)
                    break
                elif k == 'D':
                    found_back = ('D', v)
                    break  
                elif k == 'F':
                    found_back = ('F', v)
                    break

            found_forward = None 
            for j_forward in range(i+1, len(phase_events)):
                k_fwd, v_fwd = next(iter(phase_events[j_forward].items()))
                if k_fwd == 'D':
                    found_forward = ('D', v_fwd)
                    break
                elif k_fwd == 'F':
                    found_forward = ('F', v_fwd)
                    break
                elif k_fwd.startswith('start_') or k_fwd.startswith('end_'):
                    break

            # Case 1 : T after start_i - T during real tic (excel)
            if found_back is not None and found_back[0] == 'start_i':
                results_list.append({
                    "type": "start_then_T",
                    "start_time": found_back[1]  
                })
                print(f"start_tic at {found_back[1]} then T pressed at {value}")
                print("Results list updated:", results_list)

            # Case 2 : T after end_i 
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
                    print("Results list updated:", results_list)
            
                # Case 3 : T before start_i 
                elif found_back is not None and found_back[0] == 'end_i':
                    if found_forward is not None and found_forward == 'start_i':
                        results_list.append({
                            "type": "T_before_start",
                            "T_time": value,
                        })
                        print(f"T pressed at {value} before tic starts at {found_forward[1]}")
                        print("Results list updated:", results_list)

            # Case 4 : T during the imitated tic (D before T)
            elif found_back is not None and found_back[0] == 'D':

                # Check if T comes after F (end of imitated tic)
                if found_forward is not None and found_forward[0] == 'F':
                        time_after_F = value - found_back[1] #check 
                        if time_after_F <= max_t_after_F:

                            D_found = None
                            for j in range(i-1, -1, -1):
                                k, v = next(iter(phase_events[j].items()))
                                if k == 'D':
                                    D_found = v
                                    break

                            upgraded = False
                            for res in results_list:
                                if res["type"] == "D_then_F" and res["D_time"] == D_found:
                                    res["type"] = "D_then_F_then_T"
                                    upgraded = True
                                    print(f"Upgraded D_then_F → D_then_F_then_T at D={D_found}")
                                    break

                            if not upgraded:
                                results_list.append({
                                    "type": "F_then_T",
                                    "D_time": D_found
                                })

                else:
                    # Just D then T, no F after
                    results_list.append({
                        "type": "D_then_T",
                        "D_time": found_back[1],
                    })
                    print(f"T pressed at {value} during imitated tic started at {found_back[1]}")
                    print("Results list updated:", results_list)

            # Case 5 : T after F (end of imitated tic)
            elif found_back is not None and found_back[0] == 'F':
                time_after_F = value - found_back[1]
                if time_after_F <= max_t_after_F:
                    # Find the D before this F
                    D_found = None
                    for j in range(i-1, -1, -1):
                        key_k, value_k = next(iter(phase_events[j].items()))
                        if key_k == 'D':
                            D_found = value_k
                            break  
                        
                    results_list.append({
                        "type": "F_then_T",
                        "D_time": D_found,  
                    })
                    print(f"F at {found_back[1]} then T pressed at {value} (close after imitated tic)")
                    print("Results list updated:", results_list)

                elif found_back is not None and found_back[0] == 'F':
                    if found_forward is not None and found_forward[0] == 'D':
                        results_list.append({
                            "type": "T_before_D",
                            "T_time": value,
                        })
                        print(f"T pressed at {value} before D at {found_forward[1]}")
                        print("Results list updated:", results_list)
   
        # ----------- Imitated tic -----------
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
            
            found_forward = None
            for j_forward in range(i+1, len(phase_events)):
                k_fwd, v_fwd = next(iter(phase_events[j_forward].items()))
                if k_fwd.startswith('start_'):
                    found_forward = ('start_i', v_fwd)
                    break
                elif k_fwd == 'F':
                    found_forward = ('F', v_fwd)
                    break
                elif k_fwd == 'T':
                    break
                elif k_fwd == 'D':
                    break

            # Case 1: D after start_i 
            if found_back is not None and found_back[0] == 'start_i':
                    results_list.append({
                        "type": "start_then_D",
                        "start_time": found_back[1],
                    })
                    print(f"start_tic at {found_back[1]} then D pressed at {value}")
                    print("Results list updated:", results_list)
                
            # Case 2 : D after D
            if found_back is not None and found_back[0] == 'D':
                # Look forward to see if there's start_i or just F
                found_forward = None
                for j_forward in range(i+1, len(phase_events)):
                    k_fwd, v_fwd = next(iter(phase_events[j_forward].items()))
                    if k_fwd.startswith('start_'):
                        found_forward = ('start_i', v_fwd)
                        break
                    elif k_fwd == 'F':
                        found_forward = ('F', v_fwd)
                        break
                    elif k_fwd == 'T' or k_fwd == 'D':
                        break
                
                # Exclude T in between the imitated tic 
                if found_forward is not None and found_forward[0] == 'T':
                    i += 1
                    continue

                # Only add D_then_F if there's F but NO start_i
                elif found_forward is not None and found_forward[0] == 'F':    
                
                        results_list.append({
                            "type": "D_then_F",
                            "D_time": value,
                        })
                        print(f"D at {value} then F pressed at {found_forward[1]} (no visible tic)")
                        print("Results list updated:", results_list)
                else: 
                    results_list.append({
                        "type": "D_then_D",
                        "D_time": value,  
                    })
                    print(f"D pressed at {value}")
                    print("Results list updated:", results_list)


            # Case 3: D after end_i 
            elif found_back is not None and found_back[0] == 'end_i':
                found_forward = None
                for j_forward in range(i+1, len(phase_events)):
                    k_fwd, v_fwd = next(iter(phase_events[j_forward].items()))
                    if k_fwd.startswith('start_'):
                        found_forward = ('start_i', v_fwd)
                        break
                    elif k_fwd == 'F':
                        found_forward = ('F', v_fwd)
                        break
                    elif k_fwd == 'T' or k_fwd == 'D':
                        break

                # D before start_i 
                if found_forward is not None and found_forward[0] == 'start_i':
                        results_list.append({
                            "type": "D_then_start",
                            "D_time": value,
                        })
                        print(f"D pressed at {value} then tic starts at {found_forward[1]}")
                        print("Results list updated:", results_list)
    

        i += 1

    # Filter out D_then_F entries where D was used in D_then_F_then_T
    F_then_T_D_times = {
        res["D_time"]
        for res in results_list
        if res["type"] == "F_then_T" and res.get("D_time") is not None
    }

    new_results = []
    upgraded_D_times = set()

    for res in results_list:
        if res["type"] == "D_then_F" and res["D_time"] in F_then_T_D_times
            new_results.append({
                "type": "D_then_F_then_T",
                "D_time": res["D_time"]
            })
            upgraded_D_times.add(res["D_time"])

        elif res["type"] == "F_then_T" and res["D_time"] in upgraded_D_times:
            continue

        else:
            new_results.append(res)

    results_list = new_results
    print ("Final Results List:", results_list)



    return results_list