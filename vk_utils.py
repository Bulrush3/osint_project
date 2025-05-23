import vk_api
import random

VK_TOKEN = ''

vk_session = vk_api.VkApi(token=VK_TOKEN)
vk = vk_session.get_api()


def get_group_members_count(group_id):
    """Получить количество участников группы."""
    try:
        info = vk.groups.getMembers(group_id=group_id)
        return info['count']
    except vk_api.exceptions.ApiError as e:
        print(f"Ошибка при получении количества участников группы {group_id}: {e}")
        return 0
    

def vk_get_group_members(group_id, offset=1000, count=100):
    try:
        members = vk.groups.getMembers(
            group_id=group_id,
            offset=offset,
            count=count,
            fields=['bdate', 'sex', 'city', 'country', 'deactivated', 'is_closed', 'can_access_closed', 'photo_100', 'last_seen']
        )
        # Фильтруем только живых и открытых
        filtered = []
        for u in members['items']:
            if u.get('deactivated'):
                continue
            if u.get('is_closed') and not u.get('can_access_closed'):
                continue
            filtered.append(u)
        return filtered
    except vk_api.exceptions.ApiError as e:
        print(f"Ошибка при получении участников группы {group_id}: {e}")
        return []
    
def get_user_groups(user_id):
    try:
        response = vk.users.getSubscriptions(
            user_id=user_id,
            extended=1,
            fields=['name', 'status']  # Запрашиваем name и status
        )
        
        items = response.get('items', []) # Получаем список групп, по умолчанию пустой список
        
        processed_groups = []
        for item_dict in items[:30]: # item_dict - это полный словарь группы от VK API
            # Создаем новый словарь только с нужными полями
            filtered_group_info = {
                'id': item_dict.get('id'),
                'name': item_dict.get('name'),
                'status': item_dict.get('status')
            }
            # print(filtered_group_info)
            processed_groups.append(filtered_group_info)
            
        return processed_groups # Возвращаем список отфильтрованных словарей
        
    except vk_api.exceptions.ApiError as e:
        if e.code == 30:  # Profile is private
            return None
        print(f"Ошибка при получении групп пользователя {user_id}: {e}")
        return None
    except Exception as e: # Отлов других возможных ошибок
        print(f"Неожиданная ошибка при обработке подписок для пользователя {user_id}: {e}")
        return None


def collect_alive_users_from_groups(groups, n_target=100):
    users = []
    checked = 0
    for gid in groups:
        total = get_group_members_count(gid)
        print('Обработана группа')
        offset = random.randint(0, max(0, total-1000))
        members = vk_get_group_members(gid, offset=offset, count=100)
        print('Собрано человек:', len(members))
        for u in members:
            # Проверяем, не набрали ли уже нужное количество
            if len(users) >= n_target:
                return users
            # Пробуем получить группы пользователя
            groups_info = get_user_groups(u["id"])
            checked += 1
            if groups_info is None:
                continue  # Пропускаем удалённых/закрытых/забаненных
            users.append({
                "user_id": u["id"],
                "bdate":   u.get("bdate"),
                "sex":     u.get("sex"),
                "city":    u.get("city", {}).get("title"),
                "country": u.get("country", {}).get("title"),
                "groups": groups_info
            })
    print(f"Проверено пользователей: {checked}, собрано живых: {len(users)}")
    return users