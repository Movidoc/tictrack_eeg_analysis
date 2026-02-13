#   ==============================================================================
# Function:  deduplicate tics from the imitated phase
# ==============================================================================

def deduplicate_imitated_tics(results_list):
    '''
    Remove duplicate suppressed tic entries based on dominance rules:
    1. S_then_T_then_F dominates S_then_F (same S_time)
   2. S_then_T + S_then_F (same S_time) → merge to S_then_T_then_F
    3. S_then_F_then_T dominates S_then_F (same S_time)
    4. S_then_F dominates S_then_S (same S_time)
    '''
    # Group entries by S_time for easier comparison
    D_time_groups = {}
    other_entries = []
    
    for entry in results_list:
        if 'D_time' in entry:
            D_time = entry['D_time']
            if D_time not in D_time_groups:
                D_time_groups[D_time] = []
            D_time_groups[D_time].append(entry)
        else:
            # Entries without S_time (real tics) are kept as-is
            other_entries.append(entry)
    
    deduplicated = []
    
    for D_time, entries in D_time_groups.items():

        types_present = {entry['type'] for entry in entries}
        # Rule 1 & 2: If S_then_T_then_F exists, keep only that
        if 'D_then_T_then_F' in types_present:
            deduplicated.append({
                'type': 'D_then_T_then_F',
                'D_time': D_time
            })
    
        # Rule 2: If both S_then_T and S_then_F exist, merge to S_then_T_then_F
        elif 'D_then_T' in types_present and 'D_then_F' in types_present:
            deduplicated.append({
                'type': 'D_then_T_then_F',
                'D_time': D_time
            })
        
        # Rule 3: If S_then_F_then_T exists, keep only that (dominates S_then_F)
        elif 'D_then_F_then_T' in types_present:
            deduplicated.append({
                'type': 'D_then_F_then_T',
                'D_time': D_time
            })
        
        # Rule 4: If S_then_F exists, keep it (dominates S_then_S)
        elif 'D_then_F' in types_present:
            deduplicated.append({
                'type': 'D_then_F',
                'D_time': D_time
            })
        
        # Keep S_then_T if it's the only one
        elif 'D_then_T' in types_present:
            deduplicated.append({
                'type': 'D_then_T',
                'D_time': D_time
            })
        
        # Keep S_then_S only if nothing else exists
        elif 'D_then_D' in types_present:
            deduplicated.append({
                'type': 'D_then_D',
                'D_time': D_time
            })
        
        # Any other suppressed tic types
        else:
            deduplicated.extend(entries)
    
    # Add back all non-suppressed entries (real tics)
    deduplicated.extend(other_entries)
    
    # Sort by time for readability (using S_time, T_time, or start_time)
    def get_time(entry):
        return entry.get('D_time') or entry.get('T_time') or entry.get('start_time') or 0
    
    deduplicated.sort(key=get_time)
    
    return deduplicated

# ==============================================================================
# Function: exctract imitated and real tics entries from imitated phase 
# ==============================================================================
def analyse_merged_ttl_tics_imitated(merged_ttl_tics,
                                    phase_start_key='start_spont',
                                    phase_end_key='end_spont',
                                    max_t_after_end=1.0,
                                    max_t_after_F=2.0,
                                    max_t_before_D=2.0,
                                    max_t_before_start=2.0):
                            
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

    start_used = set() 
    D_used = set()
    F_used = set()
    D_start_used = {}
    start_D_used = {}



    while i < len(phase_events):

        key, value = next(iter(phase_events[i].items()))

           
        # ----------- Imitated tic -----------
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
                elif k =='D':
                    found_back = ('D', v)
                    break
                elif k == 'F': 
                    found_back = ('F', v) 
                    break
                elif k == 'T':
                    found_back = ('T', v)
                    break

            
            found_forward = None
            for j_forward in range(i+1, len(phase_events)):
                k_fwd, v_fwd = next(iter(phase_events[j_forward].items()))
                if k_fwd == 'F':
                    found_forward = ('F', v_fwd)
                    break
                elif k_fwd == 'D':
                    found_forward = ('D', v_fwd)
                    break
                elif k_fwd.startswith('start_'):
                    found_forward = ('start_i', v_fwd)
                    break
                elif k_fwd.startswith('end_'):
                    found_forward = ('end_i', v_fwd)
                    break
                elif k_fwd == 'T':  
                    found_forward = ('T', v_fwd)
                    break

            # ----- D then D -----
            if found_forward is not None and found_forward[0] == 'D':
                # results_list.append({
                #     "type": "D_then_D",
                #     "D_time": value,
                # })
                print(f"D pressed at {value} then D pressed at {found_forward[1]} (imitated tic)")
            
            # ------ D before start ------
            elif found_forward is not None and found_forward[0] == 'start_i':
                results_list.append({
                    "type": "D_then_start",
                    "D_time": value ,
                    }) 
                start_used.add(found_forward[1])
                D_start_used[found_forward[1]] = value
                print(f"D pressed at {value} before tic starts at {found_forward[1]}")

            # ------ D after start ------
            elif found_back is not None and found_back[0] == 'start_i':
                if found_back[1] not in D_start_used:

                    results_list.append({
                        "type": "start_then_D",
                        "start_time": found_back[1],
                    })
                    start_used.add(found_back[1])
                    start_D_used[value] = found_back[1]
                    print(f"start_tic at {found_back[1]} then D pressed at {value}")

            # ------ D after T ------
            elif found_back is not None and found_back[0] == 'T':
                if value - found_back[1] <= max_t_before_D and value in D_used:
                    if found_forward is not None and found_forward[0] == 'F':
                        # Remove the T_before_S entry that was created earlier
                        results_list = [r for r in results_list if not (r.get('type') == 'T_before_D' and r.get('T_time') == found_back[1])]
                        
                        results_list.append({
                            "type": "T_then_D_then_F",
                            "T_time": found_back[1],
                        })
                        D_used.add(value)
                        print(f"T at {found_back[1]} then D at {value} then F at {found_forward[1]}")
            
            # ----- D then F -----
            elif found_forward is not None and found_forward[0] == 'F':
                if found_forward[1] not in F_used:
                    results_list.append({
                        "type": "D_then_F",
                        "D_time": value,
                    })
                    D_used.add(value)
                    print(f"D at {value} then F pressed at {found_forward[1]}")


            
            

        # ---------- Real tic -----------
        if key == 'T':

            found_back = None
            for j in range(i-1, -1, -1):
                k, v = next(iter(phase_events[j].items()))
                if k.startswith('start_'):
                    found_back = ('start_i', v)
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
                elif k =='T': 
                    found_back = ('T', v)
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
                elif k_fwd.startswith('start_'):
                    found_forward = ('start_i', v_fwd)
                    break
                elif k_fwd.startswith('end_'):
                    found_forward = ('end_i', v_fwd)
                    break
                elif k_fwd == 'T':
                    found_forward = ('T', v_fwd)
                    break

            # --- T after T ---
            if found_back is not None and found_back[0] == 'T':
                print(f"T pressed at {value} then T pressed at {found_back[1]} (possible error or multiple T presses)")
                i += 1
                continue 

            # ---- T during imitated tic ---- 
            if found_back is not None and found_back[0] == 'D':

                if found_forward is not None and found_forward[0] == 'F':
                    results_list.append({
                        "type": "D_then_T_then_F",
                        "D_time": found_back[1],
                    })
                    print (f"D at {found_back[1]} then F at {found_forward[1]} T during pressed at {value}")

                if found_back[1] in start_D_used:
                    results_list = [r for r in results_list if not (r.get('type') == 'D_then_T' and r.get('D_time') == found_back[1])]
                    results_list.append({
                        "type": "start_then_D_then_T",
                        "start_time": start_D_used[found_back[1]],
                    })
                    start_used.add(start_D_used[found_back[1]])
                    print(f"start_tic at {start_D_used[found_back[1]]} then D at {found_back[1]} then T at {value}")
                else:
                    results_list.append({
                        "type": "D_then_T",
                        "D_time": found_back[1],
                    })
                    print(f"T pressed at {value} during imitated tic started at {found_back[1]}")

                i += 1
                continue

            # ---- T before D ---- shouldnt be the case!
            if (found_forward is not None 
                and found_forward[0] == 'D'
                and (found_forward[1] - value <= max_t_before_D)
            ): 
                results_list.append({
                    "type": "T_before_D",
                    "T_time": value,
                }) 
                D_used.add(found_forward[1])
                print(f"T pressed at {value} before D at {found_forward[1]}")
                i += 1
                continue 

            # ---- T before real tic ----
            if (found_forward is not None and 
            found_forward[0] == 'start_i'):
                if found_forward[1] - value <= max_t_before_start:
                    start_used.add(found_forward[1])
                    results_list.append({
                        "type": "T_before_start",
                        "T_time": value,
                    })
                    start_used.add(found_forward[1])
                    print(f"T pressed at {value} before tic starts at {found_forward[1]}")  
                i += 1
                continue

            # ----  T during real tic ----
            if (found_back is not None and found_back[0] == 'start_i'):
                
                if found_back[1] in D_start_used:
                    # This start was part of D_then_start, upgrade to D_then_start_then_T
                    # First, remove the old D_then_start entry
                    results_list = [r for r in results_list if not (r.get('type') == 'D_then_start' and r.get('D_time') == D_start_used[found_back[1]])]
                    
                    # Now create the upgraded entry
                    results_list.append({
                        "type": "D_then_start_then_T",
                        "D_time": D_start_used[found_back[1]],
                    })
                    start_used.add(found_back[1])
                    print(f"D at {D_start_used[found_back[1]]} then tic starts at {found_back[1]} then T pressed at {value}")

                elif found_back[1] not in start_used:
                    # Regular start_then_T
                    results_list.append({
                        "type": "start_then_T",
                        "start_time": found_back[1],
                    })
                    start_used.add(found_back[1])
                    print(f"start_tic at {found_back[1]} then T pressed at {value}")

                i += 1
                continue

            # ----  T after real tic end ----
            if found_back is not None and found_back[0] == 'end_i':
                if value - found_back[1] <= max_t_after_end:
                    start_found = None
                    for j in range(i - 1, -1, -1):
                        key_k, value_k = next(iter(phase_events[j].items()))
                        if key_k.startswith('start_i'):
                            start_found = value_k
                            break
                    if start_found is not None:
                        results_list.append({
                            "type": "end_then_T",
                            "start_time": start_found,
                        })
                        print(f"end_tic at {found_back[1]} then T pressed at {value}")
                i += 1
                continue
            
            # ---- T after F ----
            if found_back is not None and found_back[0] == 'F':
                if value - found_back[1] <= max_t_after_F:
                    D_found = None
                    start_found = None
                    for j in range(i - 1, -1, -1):
                        key_k, value_k = next(iter(phase_events[j].items()))
                        if key_k == 'D':
                            D_found = value_k
                            break
                        if key_k.startswith('start_i'):
                            start_found = value_k
                            break
                    if D_found is not None:
                        if D_found in start_D_used:
                            start_time = start_D_used[D_found]

                            # Remove previous start_then_D
                            results_list = [
                                r for r in results_list
                                if not (
                                    r.get("type") == "start_then_D"
                                    and r.get("start_time") == start_time
                                )
                            ]
                            results_list.append({
                                "type": "start_then_D_then_F_then_T",
                                "start_time": start_time,
                            })
                            print(f"start at {start_time} then D at {D_found} "f"then F then T at {value}")
                        else:
                            results_list.append({
                                "type": "D_then_F_then_T",
                                "D_time": D_found,
                            })
                            print( f"F at {found_back[1]} then T pressed at {D_found} " "(close after suppressed tic)")

                    elif start_found is not None:
                        results_list.append({
                            "type": "start_then_F_then_T",
                            "start_time": start_found,
                        })
                        F_used.add(found_back[1])
                        start_used.add(start_found)
                        print(f"F at {found_back[1]} then T pressed at {start_found} (close after suppressed tic)")
                    else:
                        results_list.append({
                            "type": "F_then_T",
                            "F_time": found_back[1],
                        })
                        F_used.add(found_back[1])
                        print(f"F at {found_back[1]} then T pressed at {value} (close after suppressed tic)")
                    
                i += 1
                continue


        # ----------- Imitated tic with no D -----------

        if key == 'F' :
            found_back = None
            for j in range(i - 1, -1, -1):
                k, v = next(iter(phase_events[j].items()))
                if k.startswith('start_'):
                    found_back = ('start_i', v)
                    break
                elif k.startswith('end_'):
                    found_back = ('end_i', v)
                    break
                elif k == 'D':
                    break
                elif k == 'F':
                    break
                elif k == 'T':
                    break
            
            found_forward_n = None
            for j in range(i + 1, len(phase_events)):
                k_fwd, v_fwd = next(iter(phase_events[j].items()))
                if k_fwd == 'T':
                    found_forward_n = ('T', v_fwd)
                    break
                elif k_fwd.startswith('start_'):
                    break
                elif k_fwd.startswith('end_'):
                    break
                elif k_fwd == 'D' or k_fwd == 'F':
                    break

 
            if found_back is not None and found_back[0] == 'start_i' and found_back[1] not in start_used:
                results_list.append({
                    "type": "start_then_F",
                    "start_time": found_back[1],
                })
                start_used.add(found_back[1])
                print(f"start_tic at {found_back[1]} then F pressed at {value}")
            elif found_back is not None and found_back[0] == 'end_i':
                start_found = None  
                for j in range(i - 1, -1, -1):
                    key_k, value_k = next(iter(phase_events[j].items()))
                    if key_k.startswith('start_'):
                        start_found = value_k
                        break

                if start_found is not None and start_found not in start_used and found_forward_n is not None and found_forward_n[0] == 'T':
                        results_list.append({
                            "type": "start_end_then_F_then_T",
                            "start_time": start_found,
                        })
                        print(f"end_tic at {found_back[1]} then F at {value} (already used)  T pressed at {start_found} (close after suppressed tic)")
                elif start_found is not None and start_found not in start_used and (found_forward_n is None or found_forward_n[0] != 'T'):
                    results_list.append({
                        "type": "start_end_then_F",
                        "start_time": start_found,
                    })
        i += 1

            
    results_list = deduplicate_imitated_tics(results_list)
    print("Final Results List:", results_list)

    # divide into suppressed and real tics
    # Types that involve suppression (S and/or F) but NO real tic detection (T)
    imitated_types = {
        'D_then_F',
        'D_then_D',
        'D_then_start',
        'start_then_D',
        'start_then_F',
        'start_end_then_F',
    }

    # Everything else involves a T press = real tic detection
    imitated_tics = [r for r in results_list if r['type'] in imitated_types]
    real_tics       = [r for r in results_list if r['type'] not in imitated_types]

    print("Imitated tics:", imitated_tics)
    print("Real tics:", real_tics)

    return imitated_tics, real_tics
    

# ==============================================================================
# Funtion: deduplicate suppressed and real tics from the suppressed phase 
# ==============================================================================
def deduplicate_suppressed_tics(results_list):
    '''
    Remove duplicate suppressed tic entries based on dominance rules:
    1. S_then_T_then_F dominates S_then_F (same S_time)
   2. S_then_T + S_then_F (same S_time) → merge to S_then_T_then_F
    3. S_then_F_then_T dominates S_then_F (same S_time)
    4. S_then_F dominates S_then_S (same S_time)
    '''
    # Group entries by S_time for easier comparison
    s_time_groups = {}
    other_entries = []
    
    for entry in results_list:
        if 'S_time' in entry:
            s_time = entry['S_time']
            if s_time not in s_time_groups:
                s_time_groups[s_time] = []
            s_time_groups[s_time].append(entry)
        else:
            # Entries without S_time (real tics) are kept as-is
            other_entries.append(entry)
    
    # Process each S_time group
    deduplicated = []
    
    for s_time, entries in s_time_groups.items():
        # Check what types exist for this S_time
        types_present = {entry['type'] for entry in entries}
        
        # Rule 1 & 2: If S_then_T_then_F exists, keep only that
        if 'S_then_T_then_F' in types_present:
            deduplicated.append({
                'type': 'S_then_T_then_F',
                'S_time': s_time
            })
        
        # Rule 2: If both S_then_T and S_then_F exist, merge to S_then_T_then_F
        elif 'S_then_T' in types_present and 'S_then_F' in types_present:
            deduplicated.append({
                'type': 'S_then_T_then_F',
                'S_time': s_time
            })
        
        # Rule 3: If S_then_F_then_T exists, keep only that (dominates S_then_F)
        elif 'S_then_F_then_T' in types_present:
            deduplicated.append({
                'type': 'S_then_F_then_T',
                'S_time': s_time
            })
        
        # Rule 4: If S_then_F exists, keep it (dominates S_then_S)
        elif 'S_then_F' in types_present:
            deduplicated.append({
                'type': 'S_then_F',
                'S_time': s_time
            })
        
        # Keep S_then_T if it's the only one
        elif 'S_then_T' in types_present:
            deduplicated.append({
                'type': 'S_then_T',
                'S_time': s_time
            })
        
        # Keep S_then_S only if nothing else exists
        elif 'S_then_S' in types_present:
            deduplicated.append({
                'type': 'S_then_S',
                'S_time': s_time
            })
        
        # Any other suppressed tic types
        else:
            deduplicated.extend(entries)
    
    # Add back all non-suppressed entries (real tics)
    deduplicated.extend(other_entries)
    
    # Sort by time for readability (using S_time, T_time, or start_time)
    def get_time(entry):
        return entry.get('S_time') or entry.get('T_time') or entry.get('start_time') or 0
    
    deduplicated.sort(key=get_time)
    
    return deduplicated

# =============================================================================
# Function: extract suppressed and real tics entries from the suppressed phase
# =============================================================================
def analyse_merged_ttl_tics_suppressed(
    merged_ttl_tics,
    phase_start_key='start_ret',
    phase_end_key='end_ret',
    max_t_after_end=2.0,
    max_t_after_F=2.0,
    max_t_before_start=3.0,
    max_t_before_S=2.0
):
    """
    Analyse suppressed tics.
    Episode-level logic with dominance rules:
    - Suppressed episodes (S...F) dominate real tics
    - T_before_start has priority over start_then_T
    - Only one T per episode
    - S_then_F is upgraded to S_then_T_then_F if a T occurs inside
    """

    # Phase boundaries
    t_start = next((list(d.values())[0] for d in merged_ttl_tics
                    if list(d.keys())[0] == phase_start_key), None)
    t_end = next((list(d.values())[0] for d in merged_ttl_tics
                  if list(d.keys())[0] == phase_end_key), None)

    phase_events = [
        d for d in merged_ttl_tics
        if t_start < list(d.values())[0] < t_end
    ]

    results_list = []
    i = 0


    start_used = set()          # real tic starts already paired with a T
    F_used = set()          # F keys already paired with a T
    S_used = set()          # S keys already paired
    #suppressed_active = False  # inside S...F window
    S_start_used = {}     # S keys already used 
    start_S_used = {}     # start_i keys already used in start_then_S

    while i < len(phase_events):

        key, value = next(iter(phase_events[i].items()))

        # if key == 'S':
        #     suppressed_active = True
        # elif key == 'F':
        #     suppressed_active = False



                # -------------- Suppressed tic -------------
        if key == 'S':
            found_back = None
            for j in range(i - 1, -1, -1):
                k, v = next(iter(phase_events[j].items()))
                if k.startswith('start_'):
                    found_back = ('start_i', v)
                    break
                elif k.startswith('end_'):
                    found_back = ('end_i', v)
                    break
                elif k == 'S':
                    found_back = ('S', v)
                    break
                elif k == 'F':
                    found_back = ('F', v)
                    break
                elif k == 'T':
                    found_back = ('T', v)
                    break



            found_forward = None
            for j in range(i + 1, len(phase_events)):
                k_fwd, v_fwd = next(iter(phase_events[j].items()))
                if k_fwd == 'F':
                    found_forward = ('F', v_fwd)
                    break
                elif k_fwd == 'S':
                    found_forward = ('S', v_fwd)
                    break
                elif k_fwd.startswith('start_'):
                    found_forward = ('start_i', v_fwd)
                    break
                elif k_fwd.startswith('end_'):
                    found_forward = ('end_i', v_fwd)
                    break
                elif k_fwd == 'T':  
                    found_forward = ('T', v_fwd)
                    break
            

            
            # ----- S then S -----
            if found_forward is not None and found_forward[0] == 'S':
                # results_list.append({
                #     "type": "S_then_S",
                #     "S_time": value,
                # })
                print(f"S pressed at {found_forward[1]}")

            # ------ S before start ------
            elif found_forward is not None and found_forward[0] == 'start_i':
                results_list.append({
                    "type": "S_then_start",
                    "S_time": value,
                })
                start_used.add(found_forward[1])
                S_start_used[found_forward[1]] = value
                print(f"S pressed at {value} then tic starts at {found_forward[1]}")

            # ----- S after start -----
            elif found_back is not None and found_back[0] == 'start_i':
                if found_back[1] not in S_start_used:
                    results_list.append({
                        "type": "start_then_S",
                        "start_time": found_back[1],
                    })
                    start_used.add(found_back[1])
                    start_S_used[value] = found_back[1]
    
                    print(f"start_tic at {found_back[1]} then S pressed at {value}")

            # ----- S after T (upgrade T_before_S to T_then_S_then_F) -----
            elif found_back is not None and found_back[0] == 'T':
                if value - found_back[1] <= max_t_before_S and value in S_used:
                    # Check if F comes after this S
                    if found_forward is not None and found_forward[0] == 'F':
                        # Remove the T_before_S entry that was created earlier
                        results_list = [r for r in results_list if not (r.get('type') == 'T_before_S' and r.get('T_time') == found_back[1])]
                        
                        results_list.append({
                            "type": "T_then_S_then_F",
                            "T_time": found_back[1],
                        })
                        S_used.add(value)
                        print(f"T at {found_back[1]} then S at {value} then F at {found_forward[1]}")

             # ----- S then F -----
            elif found_forward is not None and found_forward[0] == 'F':
                if found_forward[1] not in F_used:
                    results_list.append({
                        "type": "S_then_F",
                        "S_time": value,
                    })
                    S_used.add(value)
                    print(f"S at {value} then F pressed at {found_forward[1]}")

        # ------------ Real tic with T -----------
        if key == 'T':

            # ---- backward search ----
            found_back = None
            for j in range(i - 1, -1, -1):
                k, v = next(iter(phase_events[j].items()))
                if k.startswith('start_'):
                    found_back = ('start_i', v)
                    break
                elif k.startswith('end_'):
                    found_back = ('end_i', v)
                    break
                elif k == 'S':
                    found_back = ('S', v)
                    break
                elif k == 'F':
                    found_back = ('F', v)
                    break
                elif k == 'T':
                    found_back = ('T', v)
                    break

            # ---- forward search ----
            found_forward = None
            for j in range(i + 1, len(phase_events)):
                k_fwd, v_fwd = next(iter(phase_events[j].items()))
                if k_fwd == 'S':
                    found_forward = ('S', v_fwd)
                    break
                elif k_fwd == 'F':
                    found_forward = ('F', v_fwd)
                    break
                elif k_fwd.startswith('start_'):
                    found_forward = ('start_i', v_fwd)
                    break
                elif k_fwd.startswith('end_'):
                    found_forward = ('end_i', v_fwd)
                    break
                elif k_fwd == 'T':
                    found_forward = ('T', v_fwd)
                    break

            # --- T after T ---
            if found_back is not None and found_back[0] == 'T':
                # Just ignore this T, it's likely an error or duplicate
                print(f"Duplicate T at {value} after T at {found_back[1]}, ignoring.")
                i += 1
                continue

            # ---- T during suppressed tic (between S and F) ----
            if found_back is not None and found_back[0] == 'S':
                #suppressed_active = True  # lock until F

                if found_forward is not None and found_forward[0] == 'F':
                    results_list.append({
                        "type": "S_then_T_then_F",
                        "S_time": found_back[1],
                    })
                    print(f"S at {found_back[1]} then F at {found_forward[1]}  T during pressed at {value}")

                if found_back[1] in start_S_used:
                    # This start was part of start_then_S, upgrade to start_then_S_then_T
                    # First, remove the old start_then_S entry
                    results_list = [r for r in results_list if not (r.get('type') == 'start_then_S' and r.get('start_time') == start_S_used[found_back[1]])]
                    
                    # Now create the upgraded entry
                    results_list.append({
                        "type": "start_then_S_then_T",
                        "start_time": start_S_used[found_back[1]],
                    })
                    start_used.add(start_S_used[found_back[1]])
                    print(f"start tic at {start_S_used[found_back[1]]} then S at {found_back[1]} then T pressed at {value}")

                else:
                    results_list.append({
                        "type": "S_then_T",
                        "S_time": found_back[1],
                    })
                    print(f"T pressed at {value} during suppressed tic started at {found_back[1]}")

                i += 1
                continue

            # ---- T before S ----   shouldnt be the case!!!
            if (
                #not suppressed_active
                found_forward is not None
                and found_forward[0] == 'S'
                and (found_forward[1] - value <= max_t_before_S)
            ):
                results_list.append({
                    "type": "T_before_S",
                    "T_time": value,
                })
                S_used.add(found_forward[1])
                print(f"T pressed at {value} before S at {found_forward[1]}")
                i += 1
                continue

            # ---- T before real tic start ----
            if (#not suppressed_active and
                found_forward is not None and
                found_forward[0] == 'start_i'):
                if found_forward[1] - value <= max_t_before_start:
                    start_used.add(found_forward[1])
                    results_list.append({
                        "type": "T_before_start",
                        "T_time": value,
                    })
                    start_used.add(found_forward[1])
                    print(f"T pressed at {value} before tic starts at {found_forward[1]}")

                i += 1
                continue

            # ----  T during real tic ----
            if (found_back is not None and found_back[0] == 'start_i'):
                
                if found_back[1] in S_start_used:
                    # This start was part of S_then_start, upgrade to S_then_start_then_T
                    # First, remove the old S_then_start entry
                    results_list = [r for r in results_list if not (r.get('type') == 'S_then_start' and r.get('S_time') == S_start_used[found_back[1]])]
                    
                    # Now create the upgraded entry
                    results_list.append({
                        "type": "S_then_start_then_T",
                        "S_time": S_start_used[found_back[1]],
                    })
                    start_used.add(found_back[1])
                    print(f"S at {S_start_used[found_back[1]]} then tic starts at {found_back[1]} then T pressed at {value}")

                elif found_back[1] not in start_used:
                    # Regular start_then_T
                    results_list.append({
                        "type": "start_then_T",
                        "start_time": found_back[1],
                    })
                    start_used.add(found_back[1])
                    print(f"start_tic at {found_back[1]} then T pressed at {value}")

                i += 1
                continue

            # ----  T after real tic end ----
            if found_back is not None and found_back[0] == 'end_i':
                if value - found_back[1] <= max_t_after_end:
                    start_found = None
                    for j in range(i - 1, -1, -1):
                        key_k, value_k = next(iter(phase_events[j].items()))
                        if key_k.startswith('start_i'):
                            start_found = value_k
                            break
                    if start_found is not None:
                        results_list.append({
                            "type": "end_then_T",
                            "start_time": start_found,
                        })
                        print(f"end_tic at {found_back[1]} then T pressed at {value}")
                i += 1
                continue

            # ---- T after F ----
            if found_back is not None and found_back[0] == 'F':
                if value - found_back[1] <= max_t_after_F:
                    S_found = None
                    start_found = None
                    for j in range(i - 1, -1, -1):
                        key_k, value_k = next(iter(phase_events[j].items()))
                        if key_k == 'S':
                            S_found = value_k
                            break
                        if key_k.startswith('start_i'):
                            start_found = value_k
                            break

                    if S_found is not None:
                        results_list.append({
                            "type": "S_then_F_then_T",
                            "S_time": S_found,
                        })
                        print(f"F at {found_back[1]} then T pressed at {S_found} (close after suppressed tic)")
                    elif start_found is not None:
                        results_list.append({
                            "type": "start_then_F_then_T",
                            "start_time": start_found,
                        })
                        F_used.add(found_back[1])
                        start_used.add(start_found)
                        print(f"F at {found_back[1]} then T pressed at {start_found} (close after suppressed tic)")
                    else:
                        results_list.append({
                            "type": "F_then_T",
                            "F_time": found_back[1],
                        })
                        F_used.add(found_back[1])
                        print(f"F at {found_back[1]} then T pressed at {value} (close after suppressed tic)")
                    
                i += 1
                continue
        


            
        # ----------- Suppressed tic with no S -----------

        if key == 'F' :
            found_back = None
            for j in range(i - 1, -1, -1):
                k, v = next(iter(phase_events[j].items()))
                if k.startswith('start_'):
                    found_back = ('start_i', v)
                    break
                elif k.startswith('end_'):
                    found_back = ('end_i', v)
                    break
                elif k == 'S':
                    break
                elif k == 'F':
                    break
                elif k == 'T':
                    break
            
            found_forward_n = None
            for j in range(i + 1, len(phase_events)):
                k_fwd, v_fwd = next(iter(phase_events[j].items()))
                if k_fwd == 'T':
                    found_forward_n = ('T', v_fwd)
                    break
                elif k_fwd.startswith('start_'):
                    break
                elif k_fwd.startswith('end_'):
                    break
                elif k_fwd == 'S' or k_fwd == 'F':
                    break

 
            if found_back is not None and found_back[0] == 'start_i' and found_back[1] not in start_used:
                results_list.append({
                    "type": "start_then_F",
                    "start_time": found_back[1],
                })
                print(f"start_tic at {found_back[1]} then F pressed at {value}")
            if found_back is not None and found_back[0] == 'end_i':
                start_found = None  
                for j in range(i - 1, -1, -1):
                    key_k, value_k = next(iter(phase_events[j].items()))
                    if key_k.startswith('start_'):
                        start_found = value_k
                        break
                print(f"DEBUG: F={value}, found_forward_n={found_forward_n}, start_found={start_found}, start_used={start_used}")
                if start_found is not None and start_found not in start_used and found_forward_n is not None and found_forward_n[0] == 'T':
                        results_list.append({
                            "type": "start_end_then_F_then_T",
                            "start_time": start_found,
                        })
                        print(f"end_tic at {found_back[1]} then F at {value} (already used)  T pressed at {start_found} (close after suppressed tic)")
                elif start_found is not None and start_found not in start_used and (found_forward_n is None or found_forward_n[0] != 'T'):
                    results_list.append({
                        "type": "start_end_then_F",
                        "start_time": start_found,
                    })


        i += 1


    
    results_list = deduplicate_suppressed_tics(results_list)
    print("Final Results List:", results_list)

    # divide into suppressed and real tics
    # Types that involve suppression (S and/or F) but NO real tic detection (T)
    suppressed_types = {
        'S_then_F',
        'S_then_S',
        'S_then_start',
        'start_then_S',
        'start_then_F',
        'start_end_then_F',
    }

    # Everything else involves a T press = real tic detection
    suppressed_tics = [r for r in results_list if r['type'] in suppressed_types]
    real_tics       = [r for r in results_list if r['type'] not in suppressed_types]

    print("Suppressed tics:", suppressed_tics)
    print("Real tics:", real_tics)

    return suppressed_tics, real_tics








