#!/usr/bin/env python3
"""Generate CSV, Vietnamese report, optional Excel and plots from completed runs."""
from __future__ import annotations
import argparse,collections,csv,json,math,re,statistics
from pathlib import Path
METHODS=['milp','corrected','adaptive','memetic','full']
LABELS={'milp':'MILP compact','corrected':'Tabu sửa tính đúng','adaptive':'Tabu + LSP thích nghi',
        'memetic':'GA + Tabu + LSP','full':'GA + Tabu + LSP + ALNS chờ'}


def summarize(root):
    rows=[json.loads(p.read_text()) for p in sorted(root.glob('*/*/result.json'))]
    oracle={r['instance']:r for r in rows if r['method']=='milp' and r.get('certified_optimal')}
    for r in rows:
        r['heuristic_run']=None if r['method']=='milp' else 1
        r['cplex_random_seed']=202009243 if r['method']=='milp' else None
        if r['method']=='milp':
            r['stop_reason']='proven_optimal' if r.get('certified_optimal') else 'solver_termination'
        else:
            log=root/Path(r['instance']).stem/r['method']/'search.log'
            match=re.search(r'TS iterations\s*:\s*(\d+)',log.read_text()) if log.exists() else None
            r['tabu_iterations']=int(match[1]) if match else None
            r['stop_reason']='certified_bound' if r.get('certified_stop') else (
                'iteration_limit' if r.get('tabu_iterations')==10000000 else 'time_limit')
        optimum=oracle.get(r['instance'],{}).get('objective')
        r['certified_model_optimum']=optimum
        for field,output in [('objective','gap_to_model_optimum_pct'),('checkpoint_150','gap_150_to_model_optimum_pct')]:
            if r.get(field) is not None and optimum is not None:
                if r[field]<optimum-1e-5:
                    raise ValueError(f'Feasible objective below a certified bound: {r["instance"]}/{r["method"]}/{field}')
                r[output]=max(0.0,100*(r[field]-optimum)/optimum)
        if r.get('objective') is not None:
            r['difference_vs_paper_pct']=100*(r['objective']-r['paper_milp'])/r['paper_milp']
            r['difference_vs_paper_objective']=r['objective']-r['paper_milp']
            r['difference_vs_paper_mh_pct']=100*(r['objective']-r['paper_matheuristic'])/r['paper_matheuristic']
            r['difference_vs_paper_mh_objective']=r['objective']-r['paper_matheuristic']
        if r.get('checkpoint_150') is not None:
            r['difference_150_vs_paper_pct']=100*(r['checkpoint_150']-r['paper_milp'])/r['paper_milp']
        r['comparison_note']='Input matches author data; equivalence to published result conventions is unresolved.'
    summary=[]
    for method in METHODS:
        group=[r for r in rows if r['method']==method and r.get('status')=='verified_feasible']
        finite=lambda field:[r[field] for r in group if r.get(field) is not None]
        gaps=finite('gap_to_model_optimum_pct');gaps150=finite('gap_150_to_model_optimum_pct')
        summary.append({'method':method,'description':LABELS[method],'completed':len(group),
          'optimal_final':sum(g<=1e-5 for g in gaps),'optimal_at_150':sum(g<=1e-5 for g in gaps150),
          'mean_gap_final_pct':statistics.mean(gaps) if gaps else None,
          'mean_gap_150_pct':statistics.mean(gaps150) if gaps150 else None,
          'median_seconds':statistics.median(finite('elapsed_seconds')) if finite('elapsed_seconds') else None,
          'max_seconds':max(finite('elapsed_seconds'),default=None),
          'cp_calls':sum(finite('cp_calls')),'cache_hits':sum(finite('cache_hits')),
          'warm_reuses':sum(finite('warm_reuses')),'adaptive_refinements':sum(finite('adaptive_refinements')),
          'waiting_escapes':sum(finite('waiting_escapes'))})
    def write_csv(path,data):
        if not data:return
        fields=list(dict.fromkeys(k for r in data for k in r))
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in data)
    write_csv(root/'comparison_detailed.csv',rows);write_csv(root/'method_summary.csv',summary)
    by_case=collections.defaultdict(dict)
    for r in rows:by_case[r['instance']][r['method']]=r
    complete=len(rows)==50 and all(r.get('status')=='verified_feasible' for r in rows)
    fmt=lambda x:'—' if x is None else f'{x:.6f}'
    lines=['# Báo cáo sửa mô hình và thực nghiệm — 28/10/2026','',
      f'**Trạng thái: {len(rows)}/50 lượt đã có hồ sơ; hoàn tất đầy đủ: {complete}.**','',
      'Đã chạy MILP thực tế trên máy bằng CPLEX 22.1.1, không lấy bảng tác giả làm kết quả MILP của dự án. '
      'Mỗi lượt có file nghiệm, log, phép cộng thời gian và kiểm tra vật lý độc lập bằng Python.','',
      '**Giới hạn đối chiếu bài báo:** tọa độ, release time và cấu hình đội xe khớp dữ liệu tác giả. '
      'Tuy nhiên, objective tối ưu của mô hình tái dựng có trường hợp cao hơn và có trường hợp thấp hơn '
      'giá trị được công bố với gap 0%. Chưa xác định được nguyên nhân; không thể khẳng định chỉ do làm tròn '
      'và không được gọi các số thấp hơn là cải thiện một nghiệm tối ưu của bài báo.','',
      '## Thiết lập','',
      '- 10 instance: `U_10_0.5_Num_[1-5]` và `U_10_1.0_Num_[1-5]`, từ `Instances 2`. '
      'Đối chiếu từng tọa độ/release với archive gốc lưu tại `author_Instances.zip`.',
      '- Depot `border=(0,10)`; 2 truck, 2 drone; mỗi drone chở tối đa 2 kiện; mỗi đơn có trọng lượng 1.',
      '- Truck: Manhattan, 30 km/h. Drone: Euclidean, 60 km/h. Phục vụ 3 phút; bàn giao/chuẩn bị depot 5 phút.',
      '- Heuristic chạy `--run 1` (khởi tạo seed 1, Tabu/GA seed 137); CPLEX giữ seed mặc định 202009243. '
      'Tối đa 3600 giây/phương pháp/instance; 2 luồng solver; tối đa 8 lượt chạy đồng thời.',
      '- Mốc 150 giây được lưu trong cùng lần giải. Không dừng rồi khởi động lại solver để lấy mốc này.',
      '- Heuristic được dừng sớm khi chạm cận dưới do MILP độc lập đã chứng minh trên cùng mô hình. '
      'Cận này chỉ dùng cho điều kiện dừng, không dùng để tạo tuyến, đánh giá cha mẹ hoặc chọn bước đi.',
      '- Bản thực nghiệm còn có safety cap 10 triệu vòng Tabu. `adaptive/U_10_0.5_Num_3` '
      'chạm cap sau khoảng 2724 giây và chưa đạt optimum; không gọi đây là lượt chạy đủ 3600 giây. '
      'Bản mã hiện tại đã nâng cap mặc định lên INT_MAX để ngân sách thời gian chi phối.',
      '- Thời gian heuristic không bao gồm lần MILP chạy trước để có cận dừng. '
      'Vì vậy không diễn giải thời gian dừng sớm này là chi phí chứng minh end-to-end của một solver độc lập.',
      '- Thời gian là wall-clock trên máy chạy đồng thời nhiều lượt; không so trực tiếp tốc độ với máy tác giả.','',
      '## Các thay đổi đã thực hiện','',
      '1. Đồng nhất hai parser; hỗ trợ dữ liệu gốc 3 cột, dữ liệu 6 cột, header và depot ID 0. Không đổi dữ liệu gốc.',
      '2. Sửa lỗi 20 khách bị cấp drone capacity 10 thay vì 2; bổ sung tùy chọn cấu hình tường minh.',
      '3. Tách release time khỏi turnaround: `depart >= max(release, previous_return + 5)`; chuyến đầu sẵn sàng ở t=0.',
      '4. Dùng makespan truck về depot; phân biệt giá trị CP-SAT, giá trị kiểm tra được, trạng thái giải và fallback.',
      '5. Giữ thứ tự drone do CP-SAT chọn; lập lịch lại bằng khoảng cách gốc qua đồ thị ràng buộc thời gian chung cho cả đội xe.',
      '6. Sửa aspiration: cần một upper bound khả thi tốt hơn, không chỉ một lower bound thấp. Sửa thuộc tính tabu của bước chuyển ngược.',
      '7. Bổ sung chuyển/đổi cặp pickup–delivery. Bộ benchmark này chỉ có delivery nên thành phần theo cặp được kiểm tra chức năng, không đánh giá hiệu quả qua bảng tác giả.',
      '8. Thử cache, tái sử dụng phương án nạp, LSP phân bổ thời gian thích nghi; cache không coi FEASIBLE/UNKNOWN là chứng minh tối ưu.',
      '9. Thử lớp GA quản lý nhiều tuyến với repair cặp, local Tabu và đa dạng quần thể; thêm biến thể ALNS dựa vào thời gian chờ.',
      '10. Dùng MILP theo cung cho trường hợp delivery-only và truck capacity không chặt. Giữ mô hình theo vị trí cho pickup–delivery/capacity nhỏ.',
      '11. In và lưu `objective_calculation.md`, `raw_timeline.csv`, `earliest_timeline.csv` và `audit.json`; bổ sung bước audit trong YAML.','',
      '## So sánh các cấu hình trên cùng mô hình','',
      '| Cấu hình | Đã chạy | Đạt tối ưu cuối | Đạt tối ưu ở 150 s | Gap TB cuối (%) | Gap TB ở 150 s (%) | Trung vị thời gian (s) |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["description"]} | {r["completed"]}/10 | {r["optimal_final"]}/10 | {r["optimal_at_150"]}/10 | {fmt(r["mean_gap_final_pct"])} | {fmt(r["mean_gap_150_pct"])} | {fmt(r["median_seconds"])} |')
    lines+=['','Gap trong bảng này tính theo optimum MILP của mô hình đã tái dựng, không theo bảng công bố. '
      '“Đạt tối ưu ở 150 s” nghĩa là incumbent tại mốc đó bằng optimum được xác nhận sau đợt chạy; '
      'không có nghĩa MILP đã đóng gap trước 150 s. Mỗi cấu hình là một nhóm thay đổi; '
      'kết quả chưa tách riêng tác dụng của crossover, warm start hay cache.','',
      '## Diễn giải và đề xuất','',
      'Các lỗi release, mục tiêu và thứ tự drone phải sửa trước khi đánh giá thuật toán: nếu cùng một tuyến '
      'bị cộng sai thời gian, GA/Tabu sẽ chọn theo fitness sai và phép so với MILP không còn ý nghĩa. '
      'Kiểm tra khả thi độc lập được giữ ngoài evaluator để không dùng chính công thức cần kiểm tra làm bằng chứng duy nhất.','',
      '| Tham khảo | Ý tưởng chuyển sang dự án | Vì sao cần điều chỉnh |',
      '|---|---|---|',
      '| Pina-Pardo et al. (2024) | Master routes + LSP chung cho đội drone; cận để sàng lọc tuyến | Đây là drone tiếp tế; tất cả khách vẫn do truck phục vụ |',
      '| Mahmoudinazlou & Kwon (2024) | GA quản lý quần thể tuyến, local search tinh chỉnh, evaluator xử lý lịch | Không sao chép cách chia khách truck/drone của TSPD; drone ở đây không giao trực tiếp |',
      '| Meng et al. (2024) | Kiểm tra feasibility và thời gian chờ, phá/sửa theo cặp | Benchmark hiện không có time window hay multi-visit drone, nên không thêm các đặc tính đó vào model |',
      '| Liang et al. (2026), public preview | Sàng lọc khả thi theo release, phối hợp lịch tiếp tế | Chưa có toàn văn; không tuyên bố tái lập thuật toán, locker hoặc hai loại drone của bài này |','',
      'LSP thích nghi/cache giảm việc giải lại các tuyến giống nhau; đổi lại phải giữ rõ một giá trị là upper bound '
      'khả thi hay đã tối ưu trên lưới CP. GA giữ nhiều tuyến khác nhau và local Tabu tinh chỉnh từng tuyến; '
      'ALNS dựa vào chờ nhắm vào nơi makespan bị kéo dài vì release hoặc thiếu drone. Đây là lý do thử từng nhóm cấu hình.','',
      'Trong đợt hiện tại, `full` đạt optimum mô hình trên cả 10 instance trước 150 giây; `memetic` đạt 10/10 '
      'ở cuối ngân sách. Vì vậy nên dùng `full` làm cấu hình ứng viên cho đợt tiếp theo, giữ `adaptive` làm '
      'đối chứng gọn hơn. Chưa thay mặc định solver bằng `full` chỉ từ một seed.','',
      'Cần làm tiếp theo theo thứ tự: (1) xác minh quy ước số học/service với implementation tác giả; '
      '(2) thêm seed và bộ 15–20 khách, cùng CPU/ngân sách; (3) ablation tách cache, warm hints, GA và '
      'waiting-removal; (4) benchmark riêng PDP có tải trọng chặt, không dùng kết quả delivery-only để thay thế.','',
      '## Đối chiếu từng instance với bảng tác giả','',
      'Sai khác (%) = 100 × (objective dự án − objective công bố) / objective công bố. '
      'Cột âm chỉ là sai khác số học, chưa phải bằng chứng vượt một optimum đã công bố.','',
      'CPLEX đã đóng gap trên cả 10 mô hình tái dựng. Vì vậy tăng time limit không thể hạ tiếp '
      'objective nếu giữ nguyên mô hình này; cần đối chiếu implementation và quy ước của hai phía '
      'trước khi quy sai khác cho chất lượng bộ giải. Bước xác minh hữu ích nhất là có nghiệm/timeline '
      'hoặc LP gốc của tác giả cho instance đầu tiên có giá trị công bố 149.','',
      '| Instance | Bảng tác giả | MILP dự án | Tabu sửa | Tabu thích nghi | GA | GA + ALNS chờ |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name,methods in sorted(by_case.items()):
        ref=next(iter(methods.values()))['paper_milp']
        lines.append('| '+name+' | '+str(ref)+' | '+' | '.join(fmt(methods.get(m,{}).get('objective')) for m in METHODS)+' |')
    hard=by_case.get('U_10_0.5_Num_3.txt',{})
    if 'adaptive' in hard and 'full' in hard:
        a,f=hard['adaptive'],hard['full']
        lines+=['','## Trường hợp Tabu bị lặp tuyến','',
          f'Tại `U_10_0.5_Num_3`, adaptive gọi CP {a.get("cp_calls")} lần nhưng có '
          f'{a.get("cache_hits")} cache hits trong {fmt(a.get("elapsed_seconds"))} giây; '
          f'objective cuối {fmt(a.get("objective"))}. Cấu hình full đạt '
          f'{fmt(f.get("objective"))} sau {fmt(f.get("elapsed_seconds"))} giây. '
          'Cache tránh giải CP lặp, nhưng không tự tạo tuyến mới. Đây là lý do cần theo dõi '
          'số tuyến thăm lại và kích hoạt diversification khi tìm kiếm bị lặp; chỉ tăng thời gian '
          'LSP hoặc thêm cache chưa giải quyết được vấn đề khám phá tuyến. Kết quả này chưa tách riêng '
          'tác dụng của quần thể GA và waiting-removal.']
    example=root/'U_10_0.5_Num_1'/'milp'/'audit'/'objective_calculation.md'
    if example.exists():lines+=['','## Phép cộng nghiệm MILP tường minh — U_10_0.5_Num_1','',example.read_text().replace('# Explicit objective calculation',''),
      '![Lịch truck và drone của nghiệm đã kiểm tra](schedule_example.png)',
      'Thời gian trên hình là phút vận hành; các số nhỏ là ID khách/điểm hẹn. '
      'Đường đỏ là makespan, vùng xám là thời gian chờ của truck.']
    diagnostic=root/'diagnostics'/'truck_only'/'independent_subset_dp.json'
    if diagnostic.exists():
        d=json.loads(diagnostic.read_text())
        lines+=['','## Kiểm tra phụ về sai khác với bài báo','',
          f'Với `U_10_0.5_Num_1`, tắt drone: MILP cho {fmt(d["physical_definition"])} phút, '
          f'và một DP tập con độc lập cũng cho cùng số; bảng truck-only của tác giả ghi {d["published_truck_only"]} phút. '
          'Sai khác đã xuất hiện khi không có lập lịch drone. Các phép thử đổi service hoặc làm tròn trong '
          '`diagnostics/` chỉ dùng để truy nguyên; không thay tham số của 50 lượt chính thức để ép khớp bảng.']
    lines+=['','## Cách đọc hồ sơ','',
      '- `comparison_detailed.csv`: đầy đủ kết quả và gap theo hai mốc so sánh.',
      '- `method_summary.csv`: tổng hợp từng cấu hình.',
      '- `benchmark_results.xlsx`: bảng Excel và ví dụ lịch truck/drone nếu thư viện xuất Excel có sẵn.',
      '- `<instance>/<method>/audit/`: lịch nghiệm cuối; `audit_150/`: lịch tại mốc 150 giây.',
      '- `milp_solution.sol`: nghiệm thật CPLEX; `encoding.txt`: quyết định tuyến và chuyến drone của heuristic.',
      '- `solver_metrics.json`, `history.json`, `incumbents.csv`: trạng thái, cận, thời gian và lịch sử cải thiện.',
      '- `source_before/`, `source_after/`, `frozen_source_manifest.json`: mã trước sửa và mã/binary heuristic đóng băng cho đợt chạy.',
      '- Sau khi đóng băng có thêm audit/YAML/tài liệu và sửa peak capacity cho biến thể tải trọng chặt; '
      'bảng chính vẫn dùng đúng mã đóng băng. Xem `REPRODUCE.md`.',
      '- `validation/`: kết quả kiểm thử; `references.bib`: trích dẫn; `diagnostics/`: kiểm tra phụ tách khỏi bảng chính.','',
      '## Giới hạn kết luận','',
      '- Chỉ 1 seed và 10 instance nhỏ. Chưa đủ để kết luận phương pháp mạnh trên 15–20 khách hoặc có ý nghĩa thống kê.',
      '- Một lượt Tabu thích nghi dừng bởi iteration cap trước khi hết ngân sách; '
      'thời gian thực và lý do dừng nằm trong CSV. Không suy diễn rằng phương pháp này đã dùng đủ một giờ.',
      '- Bộ dữ liệu này không kích hoạt ràng buộc pickup–delivery và truck capacity 50. Không suy rộng kết quả sang PDP có tải trọng chặt.',
      '- Chưa tái lập được toàn bộ quy ước của chương trình tác giả; repository công khai cung cấp dữ liệu và kết quả, không cung cấp executable/solver source.',
      '- OR-Tools 9.11.4210 dùng lưới 0.0001 phút trong tìm kiếm. Nghiệm xuất ra được lập lịch và kiểm tra bằng thời gian di chuyển số thực gốc. '
      'Trạng thái OPTIMAL của LSP trên lưới không được dùng làm chứng nhận tối ưu toàn bài toán.','',
      '## Tài liệu định hướng','',
      '- [Pina-Pardo et al. (2024)](https://doi.org/10.1016/j.ejor.2024.01.045): phân chia tuyến và bài toán con tiếp tế; benchmark gốc.',
      '- [Mahmoudinazlou & Kwon (2024)](https://arxiv.org/abs/2303.00614): phân chia GA, bộ giải mã và local search; encoding của dự án được điều chỉnh cho drone tiếp tế.',
      '- [Meng, Chen & Li (2024)](https://eprints.lancs.ac.uk/id/eprint/207820/): kiểm tra khả thi và phối hợp thời gian trong truck–drone pickup–delivery.',
      '- [Liang, Chen & Du (2026)](https://doi.org/10.1016/j.ejor.2026.10.020): định hướng sàng lọc thời gian và phối hợp tiếp tế; chỉ tham khảo phần công khai, chưa có toàn văn.']
    (root/'BAO_CAO.md').write_text('\n'.join(lines)+'\n')
    return rows,summary,by_case


def excel(root,rows,summary,cases):
    from openpyxl import Workbook
    from openpyxl.styles import Font,PatternFill,Alignment
    wb=Workbook();wb.remove(wb.active)
    def sheet(name,data):
        ws=wb.create_sheet(name)
        if not data:return ws
        fields=list(dict.fromkeys(k for r in data for k in r));ws.append(fields)
        for r in data:
            values=[]
            for field in fields:
                value=r.get(field)
                if isinstance(value,(dict,list)):value=json.dumps(value,ensure_ascii=False)
                if isinstance(value,str) and value.startswith('='):value="'"+value
                values.append(value)
            ws.append(values)
        for cell in ws[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='24476B')
        ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
        for column in ws.columns:
            width=min(48,max(14,max(len(str(c.value or '')) for c in column)+2))
            ws.column_dimensions[column[0].column_letter].width=width
            for cell in column[1:]:
                if isinstance(cell.value,float):cell.number_format='0.000000'
        return ws
    sheet('Summary',summary);sheet('Detailed comparison',rows)
    wide=[]
    for name,methods in sorted(cases.items()):
        row={'instance':name,'paper_milp':next(iter(methods.values()))['paper_milp']}
        for m in METHODS:row[m]=methods.get(m,{}).get('objective')
        wide.append(row)
    sheet('By instance',wide)
    for method,label in [('milp','MILP'),('full','GA_ALNS')]:
        path=root/'U_10_0.5_Num_1'/method/'audit'/'audit.json'
        if path.exists():
            audit=json.loads(path.read_text())['earliest_fixed_decisions']
            sheet('Truck sums '+label,audit['trucks'])
            sheet('Timeline '+label,audit['timeline'])
            sheet('Drone '+label,audit['sorties'])
    note=wb.create_sheet('Read me')
    for text in ['Original inputs verified; numerical equivalence to author result conventions remains unresolved.',
                 'A negative difference versus the paper is not claimed as improving a proven published optimum.',
                 '3600 s maximum; certified early stops use a separate MILP bound; one adaptive run hit the frozen 10M iteration cap.',
                 '150-second snapshots are from the same optimization run; seed 1; 2 solver threads.',
                 'See BAO_CAO.md for limitations and per-instance audit folders.']:
        note.append([text])
    note.column_dimensions['A'].width=110
    for row in note:row[0].alignment=Alignment(wrap_text=True)
    wb.save(root/'benchmark_results.xlsx')


def plots(root,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=[r['method'] for r in summary]
    gaps=[r['mean_gap_150_pct'] or 0 for r in summary]
    fig,ax=plt.subplots(figsize=(8,4))
    bars=ax.bar(labels,gaps,color=['#486581','#8DA0CB','#66C2A5','#FC8D62','#8DAA67'])
    ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    ax.set_ylabel('Mean gap to independently certified model optimum (%)')
    ax.set_title('Checkpoint at 150 s — 10 instances, one run per configuration')
    ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(root/'gap_at_150.png',dpi=180);fig.savefig(root/'gap_at_150.svg');plt.close(fig)
    name='U_10_0.5_Num_3'; oracle=root/name/'milp'/'result.json'
    if not oracle.exists():return
    optimum=json.loads(oracle.read_text())['objective']
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for method in METHODS[1:]:
        history_path=root/name/method/'history.json'
        if not history_path.exists():continue
        hist=json.loads(history_path.read_text())['history']
        for ax,limit in zip(axes,[150,3600]):
            points=[(t,max(0,100*(v-optimum)/optimum)) for t,v in hist if t<=limit]
            if points:
                points.append((limit,points[-1][1]));ax.step(*zip(*points),where='post',label=method)
    axes[0].set_xlim(0,150);axes[1].set_xlim(0,3600)
    for ax in axes:ax.set_xlabel('Wall-clock time (s)');ax.grid(alpha=.25);ax.set_ylim(0,20)
    axes[0].set_ylabel('Gap to certified model optimum (%)');axes[1].legend()
    fig.suptitle(name+' — gap detail 0–20%; certified early stops held constant')
    fig.tight_layout();fig.savefig(root/'convergence_hard_instance.png',dpi=180);fig.savefig(root/'convergence_hard_instance.svg');plt.close(fig)

    # A standalone schedule figure makes the objective arithmetic inspectable.
    audit_path=root/'U_10_0.5_Num_1'/'milp'/'audit'/'audit.json'
    if audit_path.exists():
        from matplotlib.patches import Patch
        audit=json.loads(audit_path.read_text())['earliest_fixed_decisions']
        colors={'travel':'#547AA5','service':'#59A96A','handling':'#ED9B40',
                'waiting':'#D1D5DB','flight':'#66B5C7'}
        truck_ids=sorted({r['truck'] for r in audit['timeline']})
        drone_ids=sorted({r['id'] for r in audit['sorties']})
        labels=[f'Truck {v}' for v in truck_ids]+[f'Drone {d}' for d in drone_ids]
        positions={('truck',v):i for i,v in enumerate(truck_ids)}
        positions.update({('drone',d):len(truck_ids)+i for i,d in enumerate(drone_ids)})
        fig,ax=plt.subplots(figsize=(11,4))
        def segment(y,left,duration,kind):
            if duration>1e-8:ax.barh(y,duration,left=left,height=.42,color=colors[kind],edgecolor='white',linewidth=.35)
        for row in audit['timeline']:
            y=positions['truck',row['truck']]
            segment(y,row['arrival']-row['travel'],row['travel'],'travel')
            own=row['node'] in json.loads(row['packages'])
            order=['handling','waiting'] if row['node']==0 else (
                  ['waiting','handling','service'] if own else ['service','waiting','handling'])
            cursor=row['arrival']
            for kind in order:
                segment(y,cursor,row[kind],kind)
                if kind=='service' and row[kind]>0:
                    ax.text(cursor+row[kind]/2,y-.27,str(row['node']),ha='center',va='bottom',fontsize=8)
                cursor+=row[kind]
            if abs(cursor-row['departure'])>1e-5:raise ValueError('Plot timing decomposition differs from audited departure.')
        for row in audit['sorties']:
            y=positions['drone',row['id']];handover=5
            segment(y,row['depart'],row['arrive']-row['depart'],'flight')
            segment(y,row['arrive'],handover,'handling')
            segment(y,row['arrive']+handover,row['return']-row['arrive']-handover,'flight')
            ax.text(row['arrive']+handover/2,y-.27,str(row['node']),ha='center',va='bottom',fontsize=8)
        ax.axvline(audit['objective'],color='#B02A37',linestyle='--',linewidth=1.2)
        ax.set_yticks(range(len(labels)),labels);ax.set_ylim(len(labels)-.5,-.65)
        ax.set_xlim(left=0);ax.set_xlabel('Physical schedule time (minutes)')
        ax.set_title(f'U_10_0.5_Num_1: independently replayed MILP schedule; C_max={audit["objective"]:.2f} min')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.legend(handles=[Patch(color=c,label=k) for k,c in colors.items()],loc='upper center',bbox_to_anchor=(.5,-.18),ncol=5,frameon=False)
        fig.tight_layout();fig.savefig(root/'schedule_example.png',dpi=180);fig.savefig(root/'schedule_example.svg');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    rows,summary,cases=summarize(args.directory)
    try:excel(args.directory,rows,summary,cases)
    except ImportError:print('openpyxl unavailable: CSV and Markdown reports still generated.')
    try:plots(args.directory,summary)
    except ImportError:print('matplotlib unavailable: plots skipped.')
    print('Summarized',len(rows),'records in',args.directory)
if __name__=='__main__':main()
