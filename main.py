import tkinter as tk
from tkinter import ttk
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
from tkinter.scrolledtext import ScrolledText

#Muhammed Emin Gündüz
BFS_PATH = []

def load_data_and_create_graph(file_path):
    df = pd.read_excel(file_path)

    graph_data = {}
    metadata = {}

    for index, row in df.iterrows():
        author = row['author_name']
        coauthors = eval(row['coauthors'])
        paper_title = row['paper_title']

        if author not in graph_data:
            graph_data[author] = []
        if author not in metadata:
            metadata[author] = []
        metadata[author].append(paper_title)

        for coauthor in coauthors:
            if coauthor not in graph_data:
                graph_data[coauthor] = []
            if coauthor not in metadata:
                metadata[coauthor] = []
            if coauthor not in graph_data[author]:
                graph_data[author].append(coauthor)
            if author not in graph_data[coauthor]:
                graph_data[coauthor].append(author)

    nx_graph = nx.Graph()
    for a, neighbors in graph_data.items():
        nx_graph.add_node(a)
        for b in neighbors:
            nx_graph.add_edge(a, b)

    return graph_data, metadata, nx_graph

def bfs_shortest_path(graph_data, source, target):
    if source not in graph_data or target not in graph_data:
        return None
    visited = set([source])
    queue = deque([(source, [source])])

    while queue:
        current, path = queue.popleft()
        if current == target:
            return path
        for neighbor in graph_data[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


def single_source_shortest_paths(graph_data, source):
    if source not in graph_data:
        return {}
    visited = set([source])
    paths = {source: [source]}
    queue = deque([(source, [source])])

    while queue:
        current, path = queue.popleft()
        for neighbor in graph_data[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                paths[neighbor] = new_path
                queue.append((neighbor, new_path))
    return paths


def dfs_longest_path(graph_data, start):
    visited = set()

    def dfs_recursive(node, path):
        visited.add(node)
        longest = path
        for neighbor in graph_data[node]:
            if neighbor not in visited:
                current_path = dfs_recursive(neighbor, path + [neighbor])
                if len(current_path) > len(longest):
                    longest = current_path
        visited.remove(node)
        return longest

    return dfs_recursive(start, [start])

def on_click_ister_1():
    set_output_text("İster 1 seçildi: A ve B yazarları arasında en kısa yol aranıyor...")

    def ask_for_authors():
        top = tk.Toplevel(root)
        top.title("A ve B yazar (isim) bilgisi")

        label_a = ttk.Label(top, text="A yazar (isim):")
        label_a.pack(padx=10, pady=5)
        entry_a = ttk.Entry(top, width=25)
        entry_a.pack(padx=10, pady=5)

        label_b = ttk.Label(top, text="B yazar (isim):")
        label_b.pack(padx=10, pady=5)
        entry_b = ttk.Entry(top, width=25)
        entry_b.pack(padx=10, pady=5)

        def on_ok():
            a_author = entry_a.get().strip()
            b_author = entry_b.get().strip()
            top.destroy()
            process_path(a_author, b_author)

        ok_button = ttk.Button(top, text="OK", command=on_ok)
        ok_button.pack(pady=10)

    def process_path(a_author, b_author):
        if a_author == "" or b_author == "":
            set_output_text("Lütfen yazar isimlerini giriniz!")
            return

        path = bfs_shortest_path(graph_data, a_author, b_author)
        if not path:
            set_output_text(f"'{a_author}' ile '{b_author}' arasında yol bulunamadı veya geçersiz isim.")
        else:
            global BFS_PATH
            BFS_PATH = path[:]

            path_str = " -> ".join(path)
            set_output_text(f"En Kısa Yol (Kuyruk): {path_str}")

            fig = plt.gcf()
            ax = plt.gca()

            path_edges = list(zip(path, path[1:]))
            red_edges = nx.draw_networkx_edges(nx_graph, pos, edgelist=path_edges, edge_color='red', width=2, ax=ax)

            if isinstance(red_edges, list):
                drawn_edges.extend(red_edges)
            else:
                drawn_edges.append(red_edges)

            fig.canvas.draw()

    ask_for_authors()


def on_click_ister_2():
    set_output_text("İster 2 seçildi: A yazarı ve işbirliği yaptığı yazarlar, düğüm ağırlıklarına göre sıralanacak.")

    def ask_for_author():
        top = tk.Toplevel(root)
        top.title("A yazar ID bilgisi")

        label_a = ttk.Label(top, text="A yazar (isim):")
        label_a.pack(padx=10, pady=5)
        entry_a = ttk.Entry(top, width=25)
        entry_a.pack(padx=10, pady=5)

        def on_ok():
            a_author = entry_a.get().strip()
            top.destroy()
            process_queue(a_author)

        ok_button = ttk.Button(top, text="OK", command=on_ok)
        ok_button.pack(pady=10)

    def process_queue(a_author):
        if a_author not in graph_data:
            set_output_text("Geçersiz yazar ID veya graf içinde bulunamadı.")
            return

        coauthors = graph_data[a_author]
        coauthors_with_weight = [(co, len(metadata[co])) for co in coauthors]
        coauthors_with_weight.sort(key=lambda x: x[1], reverse=True)

        queue_log = []
        queue_data = []
        step = 0

        def snapshot_of_queue(q):
            return "[" + ", ".join(f"({n}, w={w})" for (n, w) in q) + "]"

        queue_log.append(f"Adım {step} - Başlangıçta Queue: {snapshot_of_queue(queue_data)}")

        for co, w in coauthors_with_weight:
            step += 1
            queue_data.append((co, w))
            queue_log.append(f"Adım {step} - Enqueue: ({co}, w={w}) -> Queue: {snapshot_of_queue(queue_data)}")

        while queue_data:
            step += 1
            removed = queue_data.pop(0)
            queue_log.append(f"Adım {step} - Dequeue: {removed} -> Queue: {snapshot_of_queue(queue_data)}")

        set_output_text("Kuyruk Adım Adım:\n" + "\n".join(queue_log))

    ask_for_author()


def on_click_ister_3():

    set_output_text("İster 3 seçildi: BFS yolundaki yazarlardan BST oluşturulacak ve 1 yazar silinecek.")

    if not BFS_PATH:
        set_output_text("Önce 1. İster çalıştırılıp bir BFS yolu (kuyruk) oluşturulmalıdır!")
        return

    class BSTNode:
        def __init__(self, value):
            self.value = value
            self.left = None
            self.right = None

    def insert_bst(root, value):
        if root is None:
            return BSTNode(value)
        if value < root.value:
            root.left = insert_bst(root.left, value)
        else:
            root.right = insert_bst(root.right, value)
        return root

    def delete_bst(root, value):
        if root is None:
            return None
        if value < root.value:
            root.left = delete_bst(root.left, value)
        elif value > root.value:
            root.right = delete_bst(root.right, value)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            else:
                successor_val = find_min(root.right)
                root.value = successor_val
                root.right = delete_bst(root.right, successor_val)
        return root

    def find_min(r):
        cur = r
        while cur.left is not None:
            cur = cur.left
        return cur.value

    root_bst = None
    for item in BFS_PATH:
        root_bst = insert_bst(root_bst, item)

    # Silinecek yazar ismini isteyelim
    def ask_for_author_to_delete():
        top = tk.Toplevel(root)
        top.title("BST'den Silinecek Yazar")

        label_del = ttk.Label(top, text="Silinecek Yazar (isim):")
        label_del.pack(padx=10, pady=5)
        entry_del = ttk.Entry(top, width=25)
        entry_del.pack(padx=10, pady=5)

        def on_ok_delete():
            author_to_del = entry_del.get().strip()
            top.destroy()
            perform_delete_and_draw(author_to_del)

        ok_button = ttk.Button(top, text="Sil", command=on_ok_delete)
        ok_button.pack(pady=10)

    def perform_delete_and_draw(yazar):
        nonlocal root_bst
        root_bst = delete_bst(root_bst, yazar)

        bst_window = tk.Toplevel(root)
        bst_window.title("BST Son Durumu")

        bst_graph = nx.Graph()

        def traverse_and_add(r):
            if r is None:
                return
            bst_graph.add_node(r.value)
            if r.left:
                bst_graph.add_node(r.left.value)
                bst_graph.add_edge(r.value, r.left.value)
                traverse_and_add(r.left)
            if r.right:
                bst_graph.add_node(r.right.value)
                bst_graph.add_edge(r.value, r.right.value)
                traverse_and_add(r.right)

        traverse_and_add(root_bst)

        fig, ax = plt.subplots(figsize=(8, 6))
        bst_pos = nx.spring_layout(bst_graph)  # Basit yerleşim (gerçek BST görünümü değil)
        nx.draw_networkx_nodes(bst_graph, bst_pos, node_color="lightgreen", ax=ax)
        nx.draw_networkx_labels(bst_graph, bst_pos, ax=ax)
        nx.draw_networkx_edges(bst_graph, bst_pos, ax=ax, edge_color="gray")

        canvas = FigureCanvasTkAgg(fig, master=bst_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(bst_window)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        set_output_text(f"{yazar} yazarı BST'den silindi. Yeni ağaç ayrı pencerede gösterildi.")

    ask_for_author_to_delete()


def on_click_ister_4():

    set_output_text("İster 4 seçildi: Kısa yollar hesaplanıyor...")

    def ask_for_author():
        top = tk.Toplevel(root)
        top.title("A yazar (isim)")

        label_a = ttk.Label(top, text="A yazar (isim):")
        label_a.pack(padx=10, pady=5)
        entry_a = ttk.Entry(top, width=25)
        entry_a.pack(padx=10, pady=5)

        def on_ok():
            a_author = entry_a.get().strip()
            top.destroy()
            process_paths(a_author)

        ok_button = ttk.Button(top, text="OK", command=on_ok)
        ok_button.pack(pady=10)

    def process_paths(a_author):
        if a_author not in graph_data:
            set_output_text("Bu yazar geçersiz veya graf içinde bulunamadı.")
            return

        paths = single_source_shortest_paths(graph_data, a_author)
        if not paths:
            set_output_text("Bu yazar için kısa yollar bulunamadı.")
            return

        formatted = [f"{k}: {' -> '.join(v)}" for k, v in paths.items()]
        set_output_text("\n".join(formatted))

    ask_for_author()


def on_click_ister_5():

    set_output_text("İster 5 seçildi: İşbirlikçi sayısı hesaplanıyor...")

    def ask_for_author():
        top = tk.Toplevel(root)
        top.title("A yazar (isim)")

        label_a = ttk.Label(top, text="A yazar (isim):")
        label_a.pack(padx=10, pady=5)
        entry_a = ttk.Entry(top, width=25)
        entry_a.pack(padx=10, pady=5)

        def on_ok():
            a_author = entry_a.get().strip()
            top.destroy()
            process_count(a_author)

        ok_button = ttk.Button(top, text="OK", command=on_ok)
        ok_button.pack(pady=10)

    def process_count(a_author):
        if a_author not in graph_data:
            set_output_text("Yazar bulunamadı veya graf içinde geçersiz.")
            return
        count = len(graph_data[a_author])
        set_output_text(f"{a_author} yazarının toplam işbirlikçi sayısı: {count}")

    ask_for_author()


def on_click_ister_6():

    set_output_text("İster 6 seçildi: En çok işbirliği yapan yazar belirleniyor...")

    most_collab_author = None
    max_degree = -1

    for node in graph_data:
        deg = len(graph_data[node])
        if deg > max_degree:
            max_degree = deg
            most_collab_author = node

    if most_collab_author:
        set_output_text(f"En çok işbirliği yapan yazar: {most_collab_author} ({max_degree} işbirlik)")
    else:
        set_output_text("Graf boş veya yazar bulunamadı.")


def on_click_ister_7():

    set_output_text("İster 7 seçildi: En uzun yol hesaplanıyor...")

    def ask_for_author():
        top = tk.Toplevel(root)
        top.title("Bir yazar (isim) giriniz")

        label_author = ttk.Label(top, text="Yazar (isim):")
        label_author.pack(padx=10, pady=5)
        entry_author = ttk.Entry(top, width=25)
        entry_author.pack(padx=10, pady=5)

        def on_ok():
            author_in = entry_author.get().strip()
            top.destroy()
            process_longest_path(author_in)

        ok_button = ttk.Button(top, text="OK", command=on_ok)
        ok_button.pack(pady=10)

    def process_longest_path(author_in):
        if author_in not in graph_data:
            set_output_text("Yazar bulunamadı veya geçersiz.")
            return

        longest_path = dfs_longest_path(graph_data, author_in)
        path_str = " -> ".join(longest_path)
        set_output_text(f"En uzun yol: {path_str}")

    ask_for_author()

def on_click_show_graph_info():
    num_nodes = nx_graph.number_of_nodes()
    num_edges = nx_graph.number_of_edges()
    info_str = f"Düğüm Sayısı: {num_nodes}\nKenar Sayısı: {num_edges}"
    set_output_text(info_str)


def on_click_stop():

    set_output_text("Stop butonu: Graf temizleniyor ve yeniden çiziliyor...")


    fig = plt.gcf()
    ax = plt.gca()
    ax.clear()


    global pos
    pos = nx.kamada_kawai_layout(nx_graph)


    degrees = dict(nx_graph.degree())
    node_sizes = [v * 5 for v in degrees.values()]


    nx.draw_networkx_nodes(nx_graph, pos, node_color="skyblue", node_size=node_sizes, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.3, edge_color="gray")


    fig.canvas.draw()


    drawn_edges.clear()


root = tk.Tk()
root.title("Graf Analiz Uygulaması")
root.geometry("1400x900")

left_frame = ttk.Frame(root, width=300, relief=tk.SUNKEN)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

output_label = ttk.Label(left_frame, text="Çıktılar:", font=("Arial", 14))
output_label.pack(anchor=tk.NW, padx=10, pady=10)


output_display = ScrolledText(left_frame, wrap='word', width=38, height=20)
output_display.pack(anchor=tk.NW, padx=10, pady=10, fill=tk.BOTH, expand=True)


def set_output_text(msg):

    output_display.delete("1.0", tk.END)
    output_display.insert(tk.END, msg)


center_frame = ttk.Frame(root, width=700, relief=tk.SUNKEN)
center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

graph_data, metadata, nx_graph = load_data_and_create_graph(
    r"C:\Users\user\Desktop\Prolablar\prolab3\prolab3_guncel_database.xlsx"
)


def draw_graph(canvas_frame):
    fig, ax = plt.subplots(figsize=(10, 10))
    global pos
    pos = nx.kamada_kawai_layout(nx_graph)

    degrees = dict(nx_graph.degree())
    node_sizes = [v * 5 for v in degrees.values()]

    nx.draw_networkx_nodes(nx_graph, pos, node_color="skyblue", node_size=node_sizes, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.3, edge_color="gray")

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    toolbar_frame = ttk.Frame(canvas_frame)
    toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()

    def on_zoom(event):
        base_scale = 1.2
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([
            xdata - (1 - relx) * new_width,
            xdata + relx * new_width
        ])
        ax.set_ylim([
            ydata - (1 - rely) * new_height,
            ydata + rely * new_height
        ])
        canvas.draw()

    def on_pan_press(event):
        if event.button == 1:
            canvas._pan_start = event.x, event.y, ax.get_xlim(), ax.get_ylim()

    def on_pan_motion(event):

        if not hasattr(canvas, '_pan_start') or canvas._pan_start is None:
            return

        x0, y0, xlim, ylim = canvas._pan_start

        if hasattr(canvas, '_pan_start'):
            x0, y0, xlim, ylim = canvas._pan_start
            dx = (event.x - x0) / canvas.get_tk_widget().winfo_width() * (xlim[1] - xlim[0])
            dy = (event.y - y0) / canvas.get_tk_widget().winfo_height() * (ylim[1] - ylim[0])
            ax.set_xlim([xlim[0] - dx, xlim[1] - dx])
            ax.set_ylim([ylim[0] + dy, ylim[1] + dy])
            canvas.draw()

    def on_node_click(event):
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        closest_node = None
        closest_distance = float('inf')
        for node, (nx_, ny_) in pos.items():
            distance = (nx_ - x) ** 2 + (ny_ - y) ** 2
            if distance < closest_distance:
                closest_distance = distance
                closest_node = node
        if closest_node:
            papers = metadata.get(closest_node, [])
            if not papers:
                set_output_text(f"{closest_node} yazarının makale bilgisi bulunamadı.")
            else:
                papers_str = "\n".join(papers)
                set_output_text(f"{closest_node} yazarının makaleleri:\n{papers_str}")

    canvas.mpl_connect("scroll_event", on_zoom)
    canvas.mpl_connect(
        "button_press_event",
        lambda event: (on_node_click(event) if event.dblclick else on_pan_press(event))
    )
    canvas.mpl_connect("motion_notify_event", on_pan_motion)
    canvas.mpl_connect("button_release_event", lambda event: setattr(canvas, '_pan_start', None))


draw_graph(center_frame)

right_frame = ttk.Frame(root, width=300, relief=tk.SUNKEN)
right_frame.pack(side=tk.LEFT, fill=tk.Y)

ister_label = ttk.Label(right_frame, text="İsterler:", font=("Arial", 14))
ister_label.pack(anchor=tk.NW, padx=10, pady=10)

ister_1_button = ttk.Button(right_frame, text="İster 1: En Kısa Yol", command=on_click_ister_1)
ister_1_button.pack(anchor=tk.NW, padx=10, pady=5)

ister_2_button = ttk.Button(right_frame, text="İster 2: İşbirlikçiler Sıralama", command=on_click_ister_2)
ister_2_button.pack(anchor=tk.NW, padx=10, pady=5)

ister_3_button = ttk.Button(right_frame, text="İster 3: BST Oluştur/Sil", command=on_click_ister_3)
ister_3_button.pack(anchor=tk.NW, padx=10, pady=5)

ister_4_button = ttk.Button(right_frame, text="İster 4: Kısa Yollar", command=on_click_ister_4)
ister_4_button.pack(anchor=tk.NW, padx=10, pady=5)

ister_5_button = ttk.Button(right_frame, text="İster 5: İşbirlikçi Sayısı", command=on_click_ister_5)
ister_5_button.pack(anchor=tk.NW, padx=10, pady=5)

ister_6_button = ttk.Button(right_frame, text="İster 6: En Çok İşbirlikçi", command=on_click_ister_6)
ister_6_button.pack(anchor=tk.NW, padx=10, pady=5)

ister_7_button = ttk.Button(right_frame, text="İster 7: En Uzun Yol", command=on_click_ister_7)
ister_7_button.pack(anchor=tk.NW, padx=10, pady=5)

stop_button = ttk.Button(right_frame, text="Stop / Temizle", command=on_click_stop)
stop_button.pack(anchor=tk.NW, padx=10, pady=20)

graph_info_button = ttk.Button(right_frame, text="Graf Bilgisi", command=on_click_show_graph_info)
graph_info_button.pack(anchor=tk.NW, padx=10, pady=5)

root.mainloop()
