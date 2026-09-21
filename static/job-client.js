// 作业订阅客户端：解析页与翻译页共用。
// 两个页面是各自独立的内嵌 <script>，共享逻辑必须外置，否则要维护两份。
(function (global) {
  async function* readSSE(resp) {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try { yield JSON.parse(line.slice(6)); } catch (e) { /* 忽略半截行 */ }
      }
    }
  }

  const MAX_FAILS = 10;

  const JobClient = {
    save(key, jobId) { try { localStorage.setItem(key, jobId); } catch (e) {} },
    forget(key) { try { localStorage.removeItem(key); } catch (e) {} },
    restore(key) { try { return localStorage.getItem(key); } catch (e) { return null; } },

    // 订阅作业直到收到 job_end。流意外中断而没有 job_end 就是断线，需要重连——
    // 这条区分是整个重连机制的判据。
    async attach(key, jobId, h) {
      let cursor = 0;
      let fails = 0;
      while (true) {
        let resp = null;
        try {
          resp = await fetch('/jobs/' + encodeURIComponent(jobId) + '/events?cursor=' + cursor);
        } catch (e) { /* 网络断了，走下面的退避重连 */ }

        if (resp && resp.status === 404) {   // 作业已被淘汰或服务重启过
          this.forget(key);
          if (h.onGone) h.onGone();
          return;
        }

        if (resp && resp.ok) {
          let replaying = 0;
          let ended = false;
          let progressed = 0;              // 收到过多少条“真事件”（不含 _replay）
          try {
            for await (const evt of readSSE(resp)) {
              if (evt.type === '_replay') {
                replaying = evt.count;
                if (h.onMeta) h.onMeta(evt);
                if (replaying === 0 && h.onReplayDone) h.onReplayDone();
                continue;                       // _replay 不在事件日志里，不推进 cursor
              }
              cursor = evt.i + 1;
              progressed++;
              if (h.onEvent) h.onEvent(evt, replaying > 0);
              if (replaying > 0 && --replaying === 0 && h.onReplayDone) h.onReplayDone();
              if (evt.type === 'job_end') {
                ended = true;
                this.forget(key);
                // status/error 必须透传：job_end 不带 file_id，两个页面的 handleEvent
                // 都会直接忽略它，收尾只能由 onEnd 做。丢掉 status 就意味着「取消」和
                // 「失败」在界面上与「成功」完全无法区分，在途卡片永远转下去。
                if (h.onEnd) h.onEnd(evt.status, evt.error);
              }
            }
          } catch (e) {
            // 硬断连（服务被杀、TCP reset、网络中断、休眠唤醒）是从 reader.read() 抛出的，
            // 不是从 fetch 抛出的。绝不能让它逃出这个循环——重连机制正是为这种断连而建，
            // 让它逃出去等于机制在唯一该生效的场景里失灵。cursor 已随每条事件推进，
            // 落到下面的退避重连即可精确续上。
            // 这里也会接住 h.onEvent 自身抛出的异常——不打印的话，两个页面 handleEvent
            // 里的真实 bug 会被静默吞掉，只看到莫名其妙的重连，看不出真正原因。
            console.warn('事件流中断或事件处理出错', e);
          }
          if (ended) return;
          // 只在真收到过事件时才清零失败计数——服务器接受连接又立刻断开（不算
          // fetch 失败，resp.ok 为真）如果无条件清零，MAX_FAILS 永远打不到，
          // 会变成无限重试的死循环。
          if (progressed > 0) fails = 0;
        }

        if (++fails > MAX_FAILS) { if (h.onGiveUp) h.onGiveUp(); return; }
        if (h.onReconnecting) h.onReconnecting(fails);
        await new Promise(r => setTimeout(r, Math.min(1000 * Math.pow(2, fails - 1), 10000)));
      }
    },

    async cancel(jobId) {
      try { await fetch('/jobs/' + encodeURIComponent(jobId) + '/cancel', { method: 'POST' }); }
      catch (e) {}
    }
  };

  global.JobClient = JobClient;
})(window);
