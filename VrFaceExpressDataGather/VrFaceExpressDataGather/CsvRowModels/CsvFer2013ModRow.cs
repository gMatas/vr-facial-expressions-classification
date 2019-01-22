using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VrFaceExpressDataGather.CsvRowModels
{
    class CsvFer2013ModRow
    {
        public int RefId { get; set; }
        public int Emotion { get; set; }
        public string LandmarksX { get; set; }
        public string LandmarksY { get; set; }
        public string Usage { get; set; }
    }
}
