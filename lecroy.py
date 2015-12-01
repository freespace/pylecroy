import numpy as np

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

    self.COMM_ORDER = 0
    self.COMM_ORDER             = self.read_enum(at(34))

    self.TEMPLATE_NAME          = self.read_string(at(16), 16)
    self.COMM_TYPE              = self.read_enum(at(32))
    self.WAVE_DESCRIPTOR_SIZE   = self.read_long(at(36))
    self.USER_TEXT_SIZE         = self.read_long(at(40))
    self.RES_DESC1_SIZE         = self.read_long(at(44))
    self.TRIGTIME_ARRAY_SIZE    = self.read_long(at(48))
    self.RIS_TIME_ARRAY_SIZE    = self.read_long(at(52))
    self.RES_ARRAY1_SIZE        = self.read_long(at(56))
    self.WAVE_ARRAY_1_SIZE      = self.read_long(at(60))

    self.VERTICAL_GAIN          = self.read_float(at(156))
    self.VERTICAL_OFFSET        = self.read_float(at(160))

    self.HORIZ_INTERVAL         = self.read_float(at(176))
    self.HORIZ_OFFSET           = self.read_double(at(180))


    aWAVE_ARRAY_1               = at( self.WAVE_DESCRIPTOR_SIZE +
                                      self.USER_TEXT_SIZE +
                                      self.TRIGTIME_ARRAY_SIZE)

###    print 'WAVE_DESCRIPTOR_SIZE', self.WAVE_DESCRIPTOR_SIZE
###    print 'USER_TEXT_SIZE', self.USER_TEXT_SIZE
###    print 'RES_DESC1_SIZE', self.RES_DESC1_SIZE
###    print 'TRIGTIME_ARRAY_SIZE', self.TRIGTIME_ARRAY_SIZE
###    print 'RIS_TIME_ARRAY_SIZE', self.RIS_TIME_ARRAY_SIZE
###    print 'RES_ARRAY1_SIZE', self.RES_ARRAY1_SIZE
###    print 'WAVE_ARRAY_1_SIZE', self.WAVE_ARRAY_1_SIZE

    self.WAVE_ARRAY_1 = self.read_wave_array(aWAVE_ARRAY_1)

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
  def WAVE_ARRAY_1_time(self):
    """
    A calculated array of when each sample in wave_form_1 was measured,
    based on HORIZ_OFFSET and HORIZ_INTERVAL.
    """
    tvec = np.arange(0, self.WAVE_ARRAY_1.size)
    return tvec * self.HORIZ_INTERVAL + self.HORIZ_OFFSET

  def savecsv(self, csvfname):
    x = np.reshape(self.WAVE_ARRAY_1_time, (-1, 1))
    y = np.reshape(self.WAVE_ARRAY_1, (-1, 1))

    mat = np.column_stack((x,y))
    np.savetxt(csvfname, mat, delimiter=',')

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

  def read_enum(self, addr):
    return self._read(addr, 2, 'u2')

  def read_long(self, addr):
    return self._read(addr, 4, 'i4')

  def read_float(self, addr):
    return self._read(addr, 4, 'f4')

  def read_double(self, addr):
    return self._read(addr, 8, 'f8')

  def read_string(self, addr, length):
    return self._read(addr, length, 'S%d'%(length))

  def read_wave_array(self, addr):
    self.fh.seek(addr)
    s = self.fh.read(self.WAVE_ARRAY_1_SIZE)
    nsamples = self.WAVE_ARRAY_1_SIZE
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
