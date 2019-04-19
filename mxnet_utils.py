import sys
import time
import re
import collections
import numpy as np

class Progbar(object):
    """Displays a progress bar.
    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, prefix='', width=10, verbose=1, interval=1,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._dynamic_display = True
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0
        self._prefix = prefix

    def update(self, current, values=None):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current
        now = time.time()
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                # sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = self._prefix + barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                eta = now - self._start
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta
                info = ' - %s' % eta_format

                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    avg = self._values[k][0] / max(1, self._values[k][1])
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.1e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


def count_param_gluon(hybridnet):
    total_param = 0
    print('%50s | %20s | %15s | %6s' % ('Name', 'shape (o,i,k)', 'dtype', '#param'))
    print(''.join(['-' for _ in range(100)]))
    for _, v in hybridnet.collect_params().items():
        temp = np.prod(v.shape)
        total_param += temp
        if len(v.name) > 50:
            name = '...' + v.name[-47:]
        else:
            name = v.name
        dtname = re.search("'(.*)'", str(v.dtype))
        dtname = v.dtype if dtname is None else dtname.group(1)
        print('%50s | %20s | %15s | %6d' % (name, str(v.shape), dtname, temp))

    print('Total Param Count: %d' % total_param)
    return total_param


def count_param_module(arg_params):
    total_param = 0
    print('%50s | %20s | %15s | %6s' % ('Name', 'shape (o,i,k)', 'dtype', '#param'))
    print(''.join(['-' for _ in range(100)]))
    for k, v in arg_params.items():
        temp = np.prod(v.shape)
        total_param += temp
        if len(k) > 50:
            name = '...' + k[-47:]
        else:
            name = k
        dtname = re.search("'(.*)'", str(v.dtype))
        dtname = v.dtype if dtname is None else dtname.group(1)
        print('%50s | %20s | %15s | %6d' % (name, str(v.shape), dtname, temp))

    print('Total Param Count: %d' % total_param)
    return total_param


def test_count_param():
    from gluoncv.model_zoo import get_model
    kwargs = {'classes': 10}
    model_name = "cifar_resnet20_v1"
    net = get_model(model_name, **kwargs)
    net.hybridize()
    count_param_gluon(net)

def test_probar():
    import time
    tloss = 0
    progbar = Progbar(target=100, prefix='Train - ', stateful_metrics=['loss'])
    for i in range(100):
        loss = np.random.rand()
        tloss += loss
        progbar.update(i + 1, [['loss', loss]])
        time.sleep(0.05)
    print(tloss/100)
if __name__ == '__main__':
    # test_count_param()
    test_probar()
    pass
