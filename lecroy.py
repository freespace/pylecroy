import numpy as np

def read_timetrace(filename):
  """
  Returns the time trace from the given file. Returns the time and
  voltage array, in that order.

  Both arrays are 1-D.
  """
  bwf = LecroyBinaryWaveform(filename)
  return bwf.WAVE_ARRAY_1_time, bwf.WAVE_ARRAY_1.ravel()

class LecroyBinaryWaveform(object):
  """
  Implemented according to specs at:

    http://teledynelecroy.com/doc/docview.aspx?id=5891

  Partially derived from lecroy.py from:

    http://qtwork.tudelft.nl/gitdata/users/guen/qtlabanalysis/analysis_modules/general/lecroy.py
  """

  def __init__(self, inputfilename):
    super(LecroyBinaryWaveform, self).__init__()

    self.fh = open(inputfilename)
    header = self.fh.read(50)
    self.aWAVEDESC = str.find(header, 'WAVEDESC')

    def at(offset):
      return self.aWAVEDESC + offset

    # the lecroy format says COMM_ORDER is an enum, which is a 16 bit
    # value and therefore subject to endianness. However COMM_ORDER
    # dictates the endianness! However, since the possible values are
    # either 0, which is the same in either endianness, or 0x1 or 0x7000
    # in big/small endianness, we can just check for 0. Since all read_*
    # methods needs a define endianness, we can default to 0 and not
    # worry about being wrong because of the preceding argument.

    # XXX The attribute names are important! Any attribute that is all
    # caps and does not start with '_' is considered metadata and will
    # be exported as part of the metadata property. This means it will
    # also be written to file when saving as CSV

    self.COMM_ORDER = 0
    self.COMM_ORDER             = self.read_enum(at(34))

    self.TEMPLATE_NAME          = self.read_string(at(16))
    self.COMM_TYPE              = self.read_enum(at(32))
    self._WAVE_DESCRIPTOR_SIZE   = self.read_long(at(36))
    self._USER_TEXT_SIZE         = self.read_long(at(40))
    self._RES_DESC1_SIZE         = self.read_long(at(44))
    self._TRIGTIME_ARRAY_SIZE    = self.read_long(at(48))
    self._RIS_TIME_ARRAY_SIZE    = self.read_long(at(52))
    self._RES_ARRAY1_SIZE        = self.read_long(at(56))
    self._WAVE_ARRAY_1_SIZE      = self.read_long(at(60))

    self.INSTRUMENT_NAME        = self.read_string(at(76))
    self.INSTRUMENT_NUMBER      = self.read_long(at(92))

    self.TRACE_LABEL            = self.read_string(at(96))

    self.TRIG_TIME              = self.read_timestamp(at(296))

    self.RECORD_TYPE            = self.read_record_type(at(316))
    self.PROCESSING_DONE        = self.read_processing_done(at(318))

    self.VERTICAL_GAIN          = self.read_float(at(156))
    self.VERTICAL_OFFSET        = self.read_float(at(160))

    self.HORIZ_INTERVAL         = self.read_float(at(176))
    self.HORIZ_OFFSET           = self.read_double(at(180))


    a_WAVE_ARRAY_1               = at( self._WAVE_DESCRIPTOR_SIZE +
                                      self._USER_TEXT_SIZE +
                                      self._TRIGTIME_ARRAY_SIZE)

###    print '_WAVE_DESCRIPTOR_SIZE', self._WAVE_DESCRIPTOR_SIZE
###    print '_USER_TEXT_SIZE', self._USER_TEXT_SIZE
###    print '_RES_DESC1_SIZE', self._RES_DESC1_SIZE
###    print '_TRIGTIME_ARRAY_SIZE', self._TRIGTIME_ARRAY_SIZE
###    print '_RIS_TIME_ARRAY_SIZE', self._RIS_TIME_ARRAY_SIZE
###    print '_RES_ARRAY1_SIZE', self._RES_ARRAY1_SIZE
###    print '_WAVE_ARRAY_1_SIZE', self._WAVE_ARRAY_1_SIZE

    self._WAVE_ARRAY_1 = self.read_wave_array(a_WAVE_ARRAY_1)

  @property
  def sampling_frequency(self):
    return 1/self.HORIZ_INTERVAL

  @property
  def LOFIRST(self):
    return not self.HIFIRST

  @property
  def HIFIRST(self):
    return self.COMM_ORDER == 0

  @property
  def WAVE_ARRAY_1(self):
    return self._WAVE_ARRAY_1

  @property
  def WAVE_ARRAY_1_time(self):
    """
    A calculated array of when each sample in wave_form_1 was measured,
    based on HORIZ_OFFSET and HORIZ_INTERVAL.
    """
    tvec = np.arange(0, self._WAVE_ARRAY_1.size)
    return tvec * self.HORIZ_INTERVAL + self.HORIZ_OFFSET

  @property
  def metadata(self):
    """
    Returns a dictionary of metadata information.
    """
    metadict = dict()
    for name, value in vars(self).items():
      if not name.startswith('_') and name.isupper():
        metadict[name] = getattr(self, name)

    return metadict

  def savecsv(self, csvfname):
    """
    Saves the binary waveform as CSV, with metadata as headers.

    The header line will contain the string

      "LECROY BINARY WAVEFORM EXPORT"

    All headers will be prepended with '#'
    """
    x = np.reshape(self._WAVE_ARRAY_1_time, (-1, 1))
    y = np.reshape(self._WAVE_ARRAY_1, (-1, 1))

    mat = np.column_stack((x,y))

    metadata = self.metadata
    jmeta = dict()
    for name, value in metadata.items():
      jmeta[name] = str(value)

    jmeta['EXPORTER'] = 'LECROY.PY'
    jmeta['AUTHOR'] = '@freespace'

    import json
    header = json.dumps(jmeta, sort_keys=True, indent=1)

    np.savetxt(csvfname, mat, delimiter=',', header=header)

  def _make_fmt(self, fmt):
    if self.HIFIRST:
      return '>' + fmt
    else:
      return '<' + fmt

  def _read(self, addr, nbytes, fmt):
    self.fh.seek(addr)
    s = self.fh.read(nbytes)
    fmt = self._make_fmt(fmt)
    return np.fromstring(s, fmt)[0]

  def read_byte(self, addr):
    return self._read(addr, 1, 'u1')

  def read_word(self, addr):
    return self._read(addr, 2, 'i2')

  def read_enum(self, addr):
    return self._read(addr, 2, 'u2')

  def read_long(self, addr):
    return self._read(addr, 4, 'i4')

  def read_float(self, addr):
    return self._read(addr, 4, 'f4')

  def read_double(self, addr):
    return self._read(addr, 8, 'f8')

  def read_string(self, addr, length=16):
    return self._read(addr, length, 'S%d'%(length))

  def read_timestamp(self, addr):
    second  = self.read_double(addr)
    addr += 8 # double is 64 bits = 8 bytes

    minute  = self.read_byte(addr)
    addr   += 1

    hour    = self.read_byte(addr)
    addr   += 1

    day     = self.read_byte(addr)
    addr   += 1

    month   = self.read_byte(addr)
    addr   += 1

    year    = self.read_word(addr)
    addr   += 2

    from datetime import datetime
    s = int(second)
    ms = int((second - s) * 1000)
    return datetime(year, month, day, hour, minute, s, ms)

  def read_processing_done(self, addr):
    v = self.read_enum(addr)
    processsing_desc = ['no_processing',
                        'fir_filter',
                        'interpolated',
                        'sparsed',
                        'autoscaled',
                        'no_result',
                        'rolling',
                        'cumulative']
    return processsing_desc[v]

  def read_record_type(self, addr):
    v = self.read_enum(addr)
    record_types = ['single_sweep',
                    'interleaved',
                    'histogram',
                    'graph',
                    'filter_coefficient',
                    'complex',
                    'extrema',
                    'sequence_obsolete',
                    'centered_RIS',
                    'peak_detect']
    return record_types[v]

  def read_wave_array(self, addr):
    self.fh.seek(addr)
    s = self.fh.read(self._WAVE_ARRAY_1_SIZE)
    nsamples = self._WAVE_ARRAY_1_SIZE
    if self.COMM_TYPE == 0:
      fmt = self._make_fmt('i1')
    else:
      fmt = self._make_fmt('i2')
      # if each sample is a 2 bytes, then we have
      # half as many samples as there are bytes in the wave
      # array
      nsamples /= 2
    dt = np.dtype((fmt, nsamples))
    data = np.fromstring(s, dtype=dt)

    # as per documentation, the actual value is gain * data - offset
    return self.VERTICAL_GAIN * data - self.VERTICAL_OFFSET

if __name__ == '__main__':
  import sys
  fname = sys.argv[1]
  bwf = LecroyBinaryWaveform(fname)

  print 'sampling freq=',bwf.sampling_frequency/1e6, 'MHz'

  bwf.savecsv(sys.argv[2])
