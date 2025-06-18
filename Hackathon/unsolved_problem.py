def analyze_organization_hierarchy(employees: pd.DataFrame) -> pd.DataFrame:
    hash_map_size = dict({})
    hash_map_level = dict({})
    hash_map_budget = dict({})

    all_ceo = employees.loc[employees["manager_id"].isnull(), "employee_id"].values

    def call_employee(id, level):
        current_total_emp = len(employees.loc[employees["manager_id"] == id,:])

        if current_total_emp == 0 and level != 1:
            hash_map_level[id] = level + 1
            hash_map_size[id] = len(employees.loc[employees["manager_id"] == id,:])
            hash_map_budget[id] = employees.loc[employees["employee_id"] == id, "salary"].values

            return len(employees.loc[employees["manager_id"] == id,:]), employees.loc[employees["employee_id"] == id, "salary"].values

        else:

            for id_emp in employees.loc[employees["manager_id"] == id, "employee_id"]:
                
                current_level = level + 1
                current_total_emp = len(employees.loc[employees["manager_id"] == id_emp,:])
                current_total_budget = employees.loc[employees["employee_id"] == id_emp, "salary"].values

                prev_total_emp, prev_total_budget = call_employee(id_emp, current_level)

                total_emp = current_total_emp + prev_total_emp
                total_budget = current_total_budget +  prev_total_budget

                if not id_emp in hash_map_size or not id_emp in hash_map_budget:
                    hash_map_level[id_emp] = current_level
                    hash_map_size[id_emp] = 0
                    hash_map_budget[id_emp] = 0

                hash_map_size[id_emp] = total_emp
                hash_map_budget[id_emp] = total_budget

                return total_emp, total_budget
        
    for ceo in all_ceo:
        prev_total_emp, prev_total_budget = call_employee(ceo, 1)
        hash_map_level[ceo] = 1
        hash_map_size[ceo] = len(employees.loc[employees["manager_id"] == ceo,:]) + prev_total_emp
        hash_map_budget[ceo] = employees.loc[employees["employee_id"] == ceo, "salary"].values + prev_total_budget


    print(hash_map_level)
    print(hash_map_size)
