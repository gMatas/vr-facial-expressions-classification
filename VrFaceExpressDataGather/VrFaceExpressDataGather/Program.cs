using System;
using System.IO;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Face;

using CsvHelper;
using VrFaceExpressDataGather.CsvRowModels;

namespace VrFaceExpressDataGather
{
    class Program
    {
        static public readonly bool runWithCsvDataset = true;
        static public readonly bool normalise = true;
        static public readonly bool localiseFace = true;
        static public readonly bool verbose = true;

        static void Main(string[] args)
        {
            // Work with images folder configuration options
            string imagesFolder = "D:\\My Work\\VR\\images";
            string outputFilepath = "D:\\My Work\\VR\\test-data-norm.txt";

            // Work with csv datasets configuration options
            Size imageSize = new Size(350, 350);
            string csvInput = "D:\\My Work\\VR\\dataset\\facial_emotions_2_ready.csv";
            string csvOutput = "D:\\My Work\\VR\\dataset\\faceexpress_dataset_v2_2.csv";

            // Models
            string faceDetectorModel = "D:\\My Work\\VR\\resources\\haarcascade_frontalface_alt2.xml";
            string facemarkModel = "D:\\My Work\\VR\\resources\\lbfmodel.yaml";

            CascadeClassifier faceDetector = new CascadeClassifier(faceDetectorModel);
            FacemarkLBFParams facemarkParams = new FacemarkLBFParams();
            FacemarkLBF facemark = new FacemarkLBF(facemarkParams);
            facemark.LoadModel(facemarkModel);

            if (runWithCsvDataset)
                RunWithCsv(faceDetector, facemark, csvInput, csvOutput, imageSize);
            else
                RunWithImagesFolder(imagesFolder, outputFilepath, faceDetector, facemark);

            Console.WriteLine("Program finished successfully!");
        }

        static private void RunWithCsv(CascadeClassifier faceDetector, FacemarkLBF facemark, string inputFilepath, string outputFilepath, Size imageSize)
        {
            using (var csvreader = new CsvReader(new StreamReader(inputFilepath)))
            using (var csvwriter = new CsvWriter(new StreamWriter(outputFilepath, false)))
            {
                csvwriter.WriteHeader<CsvFer2013ModRow>();
                csvwriter.NextRecord();

                var record = new CsvFer2013Row();
                var records = csvreader.EnumerateRecords(record);
                int recordId = 0;
                foreach (var r in records)
                {
                    recordId++;

                    Image<Gray, byte> image = StringToImage(r.pixels, imageSize);

                    Rectangle face = image.ROI;
                    if (localiseFace)
                    {
                        Rectangle? detectionResult = DetectFace(faceDetector, image);
                        if (!detectionResult.HasValue)
                            continue;
                        face = detectionResult.Value;
                    }

                    //Image<Bgr, byte> colorImage = image.Convert<Bgr, byte>();
                    //CvInvoke.Imshow("image", colorImage);
                    //CvInvoke.WaitKey();

                    VectorOfPointF landmarks = MarkFacialPoints(facemark, image, face, out bool isSuccess);
                    if (!isSuccess)
                        continue;

                    //FaceInvoke.DrawFacemarks(colorImage, landmarks, new Bgr(0, 0, 255).MCvScalar);
                    //CvInvoke.Imshow("landmarked image", colorImage);
                    //CvInvoke.WaitKey();
                    //CvInvoke.DestroyAllWindows();

                    PointF[] facepoints = landmarks.ToArray();
                    if (normalise) NormalizeFacepoints(facepoints);

                    SerializeFacepointsWithCsv(csvwriter, r, recordId, ref facepoints);

                    if (verbose) Console.Write("\rRecord No: {0}", recordId);
                }
                if (verbose) Console.WriteLine();
            }
        }

        static private void RunWithImagesFolder(string imagesFolder, string outputFilepath, CascadeClassifier faceDetector, FacemarkLBF facemark)
        {
            using (StreamWriter writer = new StreamWriter(outputFilepath, false))
            foreach (string filename in Directory.EnumerateFiles(imagesFolder))
            {
                Image<Gray, byte> image = new Image<Gray, byte>(
                    CvInvoke.Imread(filename, ImreadModes.Grayscale).Bitmap);

                Rectangle face = image.ROI;
                if (localiseFace)
                {
                    Rectangle? detectionResult = DetectFace(faceDetector, image);
                    if (!detectionResult.HasValue)
                        continue;
                    face = detectionResult.Value;
                }

                VectorOfPointF landmarks = MarkFacialPoints(facemark, image, face, out bool isSuccess);
                if (!isSuccess)
                    continue;

                PointF[] facepoints = landmarks.ToArray();
                if (normalise) NormalizeFacepoints(facepoints);

                SerializeFacepoints(writer, filename, ref facepoints);
            }
        }

        static private void SerializeFacepoints(StreamWriter writer, string filename, ref PointF[] facepoints)
        {
            writer.WriteLine("--face--");
            writer.WriteLine(filename);

            for (int i = 0; i < facepoints.Length; i++)
            {
                PointF point = facepoints[i];
                string pointStr = string.Format("{0} {1} {2}", i, point.X, point.Y);
                writer.WriteLine(pointStr);
            }
            writer.WriteLine("--end--");
            writer.WriteLine();
        }

        static private void SerializeFacepointsWithCsv(CsvWriter csvwriter, CsvFer2013Row fer2013record, int refId, ref PointF[] facepoints)
        {
            csvwriter.WriteRecord(new CsvFer2013ModRow()
            {
                RefId = refId,
                Emotion = fer2013record.emotion,
                LandmarksX = ArrayToString(from point in facepoints select point.X),
                LandmarksY = ArrayToString(from point in facepoints select point.Y),
                Usage = fer2013record.Usage
            });
            csvwriter.NextRecord();
        }

        static private Rectangle? DetectFace(CascadeClassifier detector, Image<Gray, byte> image)
        {
            Rectangle[] facesRects = detector.DetectMultiScale(image);

            if (facesRects.Length == 0)
                return null;

            // Get the biggest face rectangle
            Rectangle faceRect = facesRects[0];
            int rectEdgeLength = faceRect.Width.CompareTo(faceRect.Height) < 0 ? faceRect.Width : faceRect.Height;
            for (int i = 1; i < facesRects.Length; i++)
            {
                Rectangle rect = facesRects[i];
                int length = rect.Width.CompareTo(rect.Height) < 0 ? rect.Width : rect.Height;
                if (length > rectEdgeLength)
                {
                    faceRect = rect;
                    rectEdgeLength = length;
                }
            }
            return faceRect;
        }

        static private VectorOfPointF MarkFacialPoints(FacemarkLBF facemark, Image<Gray, byte> image, Rectangle faceRect, out bool isSuccess)
        {
            VectorOfVectorOfPointF landmarks = new VectorOfVectorOfPointF();
            VectorOfRect faces = new VectorOfRect(new Rectangle[] { faceRect });
            isSuccess = facemark.Fit(image, faces, landmarks);
            if (isSuccess)
                return landmarks[0];  // return the landmarks for the first (and only) face rectangle 
            return new VectorOfPointF();  // return an empty vector
        }

        static private void NormalizeFacepoints(PointF[] facepoints)
        {
            // Rebase facepoints to the nose point as the center of the plane.
            const int noseTipPointIndex = 30;
            PointF noseTipPoint = facepoints[noseTipPointIndex];
            PointF basePoint = new PointF(noseTipPoint.X, noseTipPoint.Y);
            for (int i = 0; i < facepoints.Length; i++)
            {
                facepoints[i].X -= basePoint.X;
                facepoints[i].Y -= basePoint.Y;
            }

            // Compute a rotation matrix to remove nose ridge tilt.
            double magnitude = Math.Sqrt(facepoints[27].X * facepoints[27].X + facepoints[27].Y * facepoints[27].Y);
            double cosAlpha = facepoints[27].X / magnitude;
            double alpha = Math.Acos(cosAlpha);
            double beta = alpha - Math.PI * .5;  // face tilt radius
            double[,] rotationMatrix = new double[,]
            {
                { Math.Cos(beta), -Math.Sin(beta) },
                { Math.Sin(beta), Math.Cos(beta) }
            };

            // Normalise the facepoints tilt using previously computed rotatation matrix.
            for (int i = 0; i < facepoints.Length; i++)
            {
                if (i == noseTipPointIndex)
                    continue;

                double x = facepoints[i].X;
                double y = facepoints[i].Y;
                double x1 = rotationMatrix[0, 0] * x + rotationMatrix[0, 1] * y;
                double y1 = rotationMatrix[1, 0] * x + rotationMatrix[1, 1] * y;
                facepoints[i].X = (float) Math.Round(x1, 9);
                facepoints[i].Y = (float) Math.Round(y1, 9);
            }

            // Locate the center of facepoints cloud.
            PointF cloudCenterPoint = new PointF(0f, 0f);
            for (int i = 0; i < facepoints.Length; i++)
            {
                cloudCenterPoint.X += facepoints[i].X;
                cloudCenterPoint.Y += facepoints[i].Y;
            }
            cloudCenterPoint.X /= facepoints.Length;
            cloudCenterPoint.Y /= facepoints.Length;

            // Rebase facepoints to the center of point cloud as the center of the plane.
            for (int i = 0; i < facepoints.Length; i++)
            {
                facepoints[i].X -= cloudCenterPoint.X;
                facepoints[i].Y -= cloudCenterPoint.Y;
            }

            // Find most deviated coordinate.
            float maxCoordDev = -1;
            foreach (PointF point in facepoints)
            {
                float xDev = Math.Abs(point.X);
                float yDev = Math.Abs(point.Y);
                float coordDev = xDev > yDev ? xDev : yDev;
                if (coordDev >= maxCoordDev)
                    maxCoordDev = coordDev;
            }

            // Normalise facepoints coordinates to fit range from -1 to 1 for both, x and y axies.
            for (int i = 0; i < facepoints.Length; i++)
            {
                facepoints[i].X /= maxCoordDev;
                facepoints[i].Y /= maxCoordDev;
            }
        }

        static private Image<Gray, byte> StringToImage(string str, Size size)
        {
            var pixels = 
                (from s in str.Trim().Split(' ')
                select Convert.ToUInt16(s)).ToArray();

            int mat1dLength = size.Height * size.Width;
            Mat mat = new Mat(1, mat1dLength, DepthType.Cv16U, 1);
            var row = mat.Row(0);
            for (int i = 0; i < mat1dLength; i++)
                row.Col(i).SetTo(new MCvScalar(pixels[i]));

            mat = mat.Reshape(0, size.Height);
            return mat.ToImage<Gray, byte>();
        }

        static private string ArrayToString<T>(IEnumerable<T> array)
        {
            return string.Join(" ", from item in array select item.ToString());
        }
    }
}
