//
//  VINDetectionViewController.m
//  TextDetection-VIN3
//
//  GitHub: https://github.com/TheLittleBoy/TextDetection-VIN3
//
//  Created by Mac on 2021/5/20.
//

#import <opencv2/opencv.hpp>       //必须要放在所有苹果API之前
#import <opencv2/imgcodecs/ios.h>
#import "ocr_db_post_process.h"
#import "ocr_crnn_process.h"
#import "paddle_api.h"
#import "OcrData.h"
#import "VINDetectionViewController.h"
#import <AVFoundation/AVFoundation.h>

using namespace paddle::lite_api;
using namespace cv;

std::shared_ptr<PaddlePredictor> net_ocr1;
std::shared_ptr<PaddlePredictor> net_ocr2;

cv::Mat resize_img_type0(const cv::Mat &img, int max_size_len, float *ratio_h, float *ratio_w) {
    int w = img.cols;
    int h = img.rows;

    float ratio = 1.f;
    int max_wh = w >= h ? w : h;
    if (max_wh > max_size_len) {
        if (h > w) {
            ratio = float(max_size_len) / float(h);
        } else {
            ratio = float(max_size_len) / float(w);
        }
    }

    int resize_h = int(float(h) * ratio);
    int resize_w = int(float(w) * ratio);
    if (resize_h % 32 == 0)
        resize_h = resize_h;
    else if (resize_h / 32 < 1)
        resize_h = 32;
    else
        resize_h = (resize_h / 32 - 1) * 32;

    if (resize_w % 32 == 0)
        resize_w = resize_w;
    else if (resize_w / 32 < 1)
        resize_w = 32;
    else
        resize_w = (resize_w / 32 - 1) * 32;

    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

    *ratio_h = float(resize_h) / float(h);
    *ratio_w = float(resize_w) / float(w);
    return resize_img;
}

void neon_mean_scale(const float *din, float *dout, int size, std::vector<float> mean, std::vector<float> scale) {
    float32x4_t vmean0 = vdupq_n_f32(mean[0]);
    float32x4_t vmean1 = vdupq_n_f32(mean[1]);
    float32x4_t vmean2 = vdupq_n_f32(mean[2]);
    float32x4_t vscale0 = vdupq_n_f32(1.f / scale[0]);
    float32x4_t vscale1 = vdupq_n_f32(1.f / scale[1]);
    float32x4_t vscale2 = vdupq_n_f32(1.f / scale[2]);

    float *dout_c0 = dout;
    float *dout_c1 = dout + size;
    float *dout_c2 = dout + size * 2;

    int i = 0;
    for (; i < size - 3; i += 4) {
        float32x4x3_t vin3 = vld3q_f32(din);
        float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
        float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
        float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
        float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
        float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
        float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
        vst1q_f32(dout_c0, vs0);
        vst1q_f32(dout_c1, vs1);
        vst1q_f32(dout_c2, vs2);

        din += 12;
        dout_c0 += 4;
        dout_c1 += 4;
        dout_c2 += 4;
    }
    for (; i < size; i++) {
        *(dout_c0++) = (*(din++) - mean[0]) / scale[0];
        *(dout_c1++) = (*(din++) - mean[1]) / scale[1];
        *(dout_c2++) = (*(din++) - mean[2]) / scale[2];
    }
}

@interface VINDetectionViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate>
{
    UILabel *textLabel;
    AVCaptureDevice *device;
    NSString *recognizedText;
    BOOL isFocus;
    BOOL isInference;
}
@property (nonatomic, assign) CGFloat m_width; //扫描框宽度
@property (nonatomic, assign) CGFloat m_higth; //扫描框高度
@property (nonatomic, strong) AVCaptureSession *session;
@property (nonatomic, strong) AVCaptureDeviceInput *videoInput;
@property (nonatomic, strong) AVCaptureVideoDataOutput *captureVideoDataOutput;
@property (nonatomic, strong) AVCaptureVideoPreviewLayer *previewLayer;

@property(nonatomic) std::vector<float> scale;
@property(nonatomic) std::vector<float> mean;
@property(nonatomic) NSArray *labels;

@end

#define SCREEN_WIDTH ([[UIScreen mainScreen] bounds].size.width)
#define SCREEN_HEIGHT ([[UIScreen mainScreen] bounds].size.height)
#define m_scanViewY  150.0
#define m_scale [UIScreen mainScreen].scale

@implementation VINDetectionViewController


- (OcrData *)paddleOcrRec:(cv::Mat)image {

    OcrData *result = [OcrData new];

    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

    cv::Mat crop_img;
    image.copyTo(crop_img);
    cv::Mat resize_img;


    float wh_ratio = float(crop_img.cols) / float(crop_img.rows);

    resize_img = crnn_resize_img(crop_img, wh_ratio);
    resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

    const float *dimg = reinterpret_cast<const float *>(resize_img.data);

    std::unique_ptr<Tensor> input_tensor0(std::move(net_ocr2->GetInput(0)));
    input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
    auto *data0 = input_tensor0->mutable_data<float>();

    neon_mean_scale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);

    //// Run CRNN predictor
    net_ocr2->Run();

    // Get output and run postprocess
    std::unique_ptr<const Tensor> output_tensor0(std::move(net_ocr2->GetOutput(0)));
    auto *rec_idx = output_tensor0->data<int>();

    auto rec_idx_lod = output_tensor0->lod();
    auto shape_out = output_tensor0->shape();
    NSMutableString *text = [[NSMutableString alloc] init];
    for (int n = int(rec_idx_lod[0][0]); n < int(rec_idx_lod[0][1] * 2); n += 2) {
        if (rec_idx[n] >= self.labels.count) {
            std::cout << "Index " << rec_idx[n] << " out of text dict range!" << std::endl;
            continue;
        }
        [text appendString:self.labels[rec_idx[n]]];
    }

    result.label = text;
    // get score
    std::unique_ptr<const Tensor> output_tensor1(std::move(net_ocr2->GetOutput(1)));
    auto *predict_batch = output_tensor1->data<float>();
    auto predict_shape = output_tensor1->shape();

    auto predict_lod = output_tensor1->lod();

    int argmax_idx;
    int blank = predict_shape[1];
    float score = 0.f;
    int count = 0;
    float max_value = 0.0f;

    for (int n = predict_lod[0][0]; n < predict_lod[0][1] - 1; n++) {
        argmax_idx = int(argmax(&predict_batch[n * predict_shape[1]], &predict_batch[(n + 1) * predict_shape[1]]));
        max_value = float(*std::max_element(&predict_batch[n * predict_shape[1]], &predict_batch[(n + 1) * predict_shape[1]]));

        if (blank - 1 - argmax_idx > 1e-5) {
            score += max_value;
            count += 1;
        }

    }
    score /= count;
    result.accuracy = score;
    return result;
}
- (NSArray *) ocr_infer:(cv::Mat) originImage{
    int max_side_len = 960;
    float ratio_h{};
    float ratio_w{};
    cv::Mat image;
    cv::cvtColor(originImage, image, cv::COLOR_RGB2BGR);

    cv::Mat img;
    image.copyTo(img);

    img = resize_img_type0(img, max_side_len, &ratio_h, &ratio_w);
    cv::Mat img_fp;
    img.convertTo(img_fp, CV_32FC3, 1.0 / 255.f);

    std::unique_ptr<Tensor> input_tensor(net_ocr1->GetInput(0));
    input_tensor->Resize({1, 3, img_fp.rows, img_fp.cols});
    auto *data0 = input_tensor->mutable_data<float>();
    const float *dimg = reinterpret_cast<const float *>(img_fp.data);
    neon_mean_scale(dimg, data0, img_fp.rows * img_fp.cols, self.mean, self.scale);
    net_ocr1->Run();
    std::unique_ptr<const Tensor> output_tensor(std::move(net_ocr1->GetOutput(0)));
    auto *outptr = output_tensor->data<float>();
    auto shape_out = output_tensor->shape();

    int64_t out_numl = 1;
    double sum = 0;
    for (auto i : shape_out) {
        out_numl *= i;
    }

    int s2 = int(shape_out[2]);
    int s3 = int(shape_out[3]);

    cv::Mat pred_map = cv::Mat::zeros(s2, s3, CV_32F);
    memcpy(pred_map.data, outptr, s2 * s3 * sizeof(float));
    cv::Mat cbuf_map;
    pred_map.convertTo(cbuf_map, CV_8UC1, 255.0f);

    const double threshold = 0.1 * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);

    auto boxes = boxes_from_bitmap(pred_map, bit_map);

    std::vector<std::vector<std::vector<int>>> filter_boxes = filter_tag_det_res(boxes, ratio_h, ratio_w, image);


    cv::Point rook_points[filter_boxes.size()][4];

    for (int n = 0; n < filter_boxes.size(); n++) {
        for (int m = 0; m < filter_boxes[0].size(); m++) {
            rook_points[n][m] = cv::Point(int(filter_boxes[n][m][0]), int(filter_boxes[n][m][1]));
        }
    }

    NSMutableArray *result = [[NSMutableArray alloc] init];

    for (int i = 0; i < filter_boxes.size(); i++) {
        cv::Mat crop_img;
        crop_img = get_rotate_crop_image(image, filter_boxes[i]);
        OcrData *r = [self paddleOcrRec:crop_img ];
        NSMutableArray *points = [NSMutableArray new];
        for (int jj = 0; jj < 4; ++jj) {
            NSValue *v = [NSValue valueWithCGPoint:CGPointMake(
                    rook_points[i][jj].x / CGFloat(originImage.cols),
                    rook_points[i][jj].y / CGFloat(originImage.rows))];
            [points addObject:v];
        }
        r.polygonPoints = points;
        [result addObject:r];
    }
    NSArray* rec_out =[[result reverseObjectEnumerator] allObjects];
    return rec_out;
}
- (NSArray *)readLabelsFromFile:(NSString *)labelFilePath {

    NSString *content = [NSString stringWithContentsOfFile:labelFilePath encoding:NSUTF8StringEncoding error:nil];
    NSArray *lines = [content componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    NSMutableArray *ret = [[NSMutableArray alloc] init];
    for (int i = 0; i < lines.count; ++i) {
        [ret addObject:@""];
    }
    NSUInteger cnt = 0;
    for (id line in lines) {
        NSString *l = [(NSString *) line stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        if ([l length] == 0)
            continue;
        NSArray *segs = [l componentsSeparatedByString:@":"];
        NSUInteger key;
        NSString *value;
        if ([segs count] != 2) {
            key = cnt;
            value = [segs[0] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        } else {
            key = [[segs[0] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] integerValue];
            value = [segs[1] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        }

        ret[key] = value;
        cnt += 1;
    }
    return [NSArray arrayWithArray:ret];
}


- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    
    self.title = @"扫一扫";
    self.view.backgroundColor = [UIColor blackColor];
    self.navigationController.navigationBar.translucent = NO;

    //
    NSString *label_file_path = [[NSBundle mainBundle] pathForResource:[NSString stringWithFormat:@"%@", @"label_list"] ofType:@"txt"];

    self.labels = [self readLabelsFromFile:label_file_path];
    self.mean = {0.485f, 0.456f, 0.406f};
    self.scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};

    NSString *model1_path = [[NSBundle mainBundle] pathForResource:[NSString stringWithFormat:@"%@", @"ch_det_mv3_db_opt"] ofType:@"nb"];
    NSString *model2_path = [[NSBundle mainBundle] pathForResource:[NSString stringWithFormat:@"%@", @"ch_rec_mv3_crnn_opt"] ofType:@"nb"];
    std::string model1_path_str = std::string([model1_path UTF8String]);
    std::string model2_path_str =  std::string([model2_path UTF8String]);
    MobileConfig config;
    config.set_model_from_file(model1_path_str);
    net_ocr1 = CreatePaddlePredictor<MobileConfig>(config);
    MobileConfig config2;
    config2.set_model_from_file(model2_path_str);
    net_ocr2 = CreatePaddlePredictor<MobileConfig>(config2);
    
    //给个默认值
    self.m_width = (SCREEN_WIDTH - 40);
    self.m_higth = 80.0;
    recognizedText = @"";
    
    //模型初始化
    [self initModel];
    
    //初始化摄像头
    [self initAVCaptureSession];
}

- (void)initModel {
}

- (void)initAVCaptureSession{
    
    self.session = [[AVCaptureSession alloc] init];
    NSError *error;
    
    device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    
    self.videoInput = [[AVCaptureDeviceInput alloc] initWithDevice:device error:&error];
    if (error) {
        NSLog(@"%@",error);
    }
    
    //输出流
    NSString* key = (NSString*)kCVPixelBufferPixelFormatTypeKey;
    NSNumber* value = [NSNumber numberWithUnsignedInt:kCVPixelFormatType_32BGRA];
    NSDictionary* videoSettings = [NSDictionary
                                   dictionaryWithObject:value forKey:key];
    self.captureVideoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
    [self.captureVideoDataOutput setVideoSettings:videoSettings];
    
    dispatch_queue_t queue;
    queue = dispatch_queue_create("cameraQueue", NULL);
    [self.captureVideoDataOutput setSampleBufferDelegate:self queue:queue];
    
    if ([self.session canAddInput:self.videoInput]) {
        [self.session addInput:self.videoInput];
    }
    if ([self.session canAddOutput:self.captureVideoDataOutput]) {
        [self.session addOutput:self.captureVideoDataOutput];
    }
    
    AVCaptureConnection* connection = [self.captureVideoDataOutput connectionWithMediaType:AVMediaTypeVideo];
    connection.videoOrientation = AVCaptureVideoOrientationPortrait;
    
    //输出照片铺满屏幕
    if ([self.session canSetSessionPreset:AVCaptureSessionPresetHigh]) {
        self.session.sessionPreset = AVCaptureSessionPresetHigh;
    }
    
    //初始化预览图层
    self.previewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:self.session];
    UIInterfaceOrientation orientation = [UIApplication sharedApplication].statusBarOrientation;
    if (orientation == UIInterfaceOrientationPortrait) {
        [[self.previewLayer connection] setVideoOrientation:AVCaptureVideoOrientationPortrait];
        
    }
    else if (orientation == UIInterfaceOrientationLandscapeLeft) {
        [[self.previewLayer connection] setVideoOrientation:AVCaptureVideoOrientationLandscapeLeft];
    }
    else if (orientation == UIInterfaceOrientationLandscapeRight) {
        [[self.previewLayer connection] setVideoOrientation:AVCaptureVideoOrientationLandscapeRight];
    }
    else {
        [[self.previewLayer connection] setVideoOrientation:AVCaptureVideoOrientationPortraitUpsideDown];
    }
    
    self.previewLayer.frame = CGRectMake(0,0, SCREEN_WIDTH,SCREEN_HEIGHT);
    
    [self.previewLayer setVideoGravity:AVLayerVideoGravityResizeAspectFill];
    
    self.view.layer.masksToBounds = YES;
    [self.view.layer addSublayer:self.previewLayer];
    
    //扫描框
    [self initScanView];
    
    //扫描结果label
    textLabel = [[UILabel alloc] initWithFrame:CGRectMake(0, (SCREEN_HEIGHT - 100)/2.0, SCREEN_WIDTH, 100)];
    textLabel.textAlignment = NSTextAlignmentCenter;
    textLabel.numberOfLines = 0;
    
    textLabel.font = [UIFont systemFontOfSize:19];
    
    textLabel.textColor = [UIColor colorWithRed:1.00 green:0.50 blue:0.00 alpha:1.00];
    [self.view addSubview:textLabel];
    
    //完成按钮
    UIButton *button = [UIButton buttonWithType:UIButtonTypeCustom];
    [self.view addSubview:button];
    button.frame = CGRectMake((SCREEN_WIDTH - 100)/2.0, SCREEN_HEIGHT - 164, 100, 50);
    [button setTitle:@"完成" forState:UIControlStateNormal];
    [button addTarget:self action:@selector(clickedFinishBtn:) forControlEvents:UIControlEventTouchUpInside];
    
    //对焦
    int flags =NSKeyValueObservingOptionNew;
    [device addObserver:self forKeyPath:@"adjustingFocus" options:flags context:nil];
}

- (void)initScanView
{
    // 中间空心洞的区域
    CGRect cutRect = CGRectMake((SCREEN_WIDTH - _m_width)/2.0,m_scanViewY, _m_width, _m_higth);
    
    UIBezierPath *path = [UIBezierPath bezierPathWithRect:CGRectMake(0,0, SCREEN_WIDTH,SCREEN_HEIGHT)];
    // 挖空心洞 显示区域
    UIBezierPath *cutRectPath = [UIBezierPath bezierPathWithRect:cutRect];
    
    //将circlePath添加到path上
    [path appendPath:cutRectPath];
    path.usesEvenOddFillRule = YES;
    
    CAShapeLayer *fillLayer = [CAShapeLayer layer];
    fillLayer.path = path.CGPath;
    fillLayer.fillRule = kCAFillRuleEvenOdd;
    fillLayer.opacity = 0.6;//透明度
    fillLayer.backgroundColor = [UIColor blackColor].CGColor;
    [self.view.layer addSublayer:fillLayer];
    
    // 边界校准线
    CGFloat lineWidth = 2;
    CGFloat lineLength = 20;
    UIBezierPath *linePath = [UIBezierPath bezierPathWithRect:CGRectMake(cutRect.origin.x - lineWidth,
                                                                         cutRect.origin.y - lineWidth,
                                                                         lineLength,
                                                                         lineWidth)];
    //追加路径
    [linePath appendPath:[UIBezierPath bezierPathWithRect:CGRectMake(cutRect.origin.x - lineWidth,
                                                                     cutRect.origin.y - lineWidth,
                                                                     lineWidth,
                                                                     lineLength)]];
    
    [linePath appendPath:[UIBezierPath bezierPathWithRect:CGRectMake(cutRect.origin.x + cutRect.size.width - lineLength + lineWidth,
                                                                     cutRect.origin.y - lineWidth,
                                                                     lineLength,
                                                                     lineWidth)]];
    [linePath appendPath:[UIBezierPath bezierPathWithRect:CGRectMake(cutRect.origin.x + cutRect.size.width ,
                                                                     cutRect.origin.y - lineWidth,
                                                                     lineWidth,
                                                                     lineLength)]];
    
    [linePath appendPath:[UIBezierPath bezierPathWithRect:CGRectMake(cutRect.origin.x - lineWidth,
                                                                     cutRect.origin.y + cutRect.size.height - lineLength + lineWidth,
                                                                     lineWidth,
                                                                     lineLength)]];
    [linePath appendPath:[UIBezierPath bezierPathWithRect:CGRectMake(cutRect.origin.x - lineWidth,
                                                                     cutRect.origin.y + cutRect.size.height,
                                                                     lineLength,
                                                                     lineWidth)]];
    
    [linePath appendPath:[UIBezierPath bezierPathWithRect:CGRectMake(cutRect.origin.x + cutRect.size.width,
                                                                     cutRect.origin.y + cutRect.size.height - lineLength + lineWidth,
                                                                     lineWidth,
                                                                     lineLength)]];
    [linePath appendPath:[UIBezierPath bezierPathWithRect:CGRectMake(cutRect.origin.x + cutRect.size.width - lineLength + lineWidth,
                                                                     cutRect.origin.y + cutRect.size.height,
                                                                     lineLength,
                                                                     lineWidth)]];
    
    CAShapeLayer *pathLayer = [CAShapeLayer layer];
    pathLayer.path = linePath.CGPath;// 从贝塞尔曲线获取到形状
    pathLayer.fillColor = [UIColor colorWithRed:0. green:0.655 blue:0.905 alpha:1.0].CGColor; // 闭环填充的颜色
    [self.view.layer addSublayer:pathLayer];
    
    UILabel *tipLabel = [[UILabel alloc] initWithFrame:CGRectMake(0, m_scanViewY - 40, SCREEN_WIDTH, 25)];
    [self.view addSubview:tipLabel];
    tipLabel.text = @"请对准VIN码进行扫描";
    tipLabel.textAlignment = NSTextAlignmentCenter;
    tipLabel.textColor = [UIColor whiteColor];
}

-(void)observeValueForKeyPath:(NSString*)keyPath ofObject:(id)object change:(NSDictionary*)change context:(void*)context {
    if([keyPath isEqualToString:@"adjustingFocus"]){
        BOOL adjustingFocus =[[change objectForKey:NSKeyValueChangeNewKey] isEqualToNumber:[NSNumber numberWithInt:1]];
        isFocus = adjustingFocus;
        NSLog(@"Is adjusting focus? %@", adjustingFocus ?@"YES":@"NO");
    }
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
    //
    if (!isFocus && !isInference) {
        isInference = YES;
        
        //得到照片
        UIImage *img = [self imageFromSamplePlanerPixelBuffer:sampleBuffer];
        
        cv::Mat originImage;
        UIImageToMat(img, originImage);
        
        //开始识别---阈值：0.8
        NSArray *data = [self ocr_infer:originImage];
        if (data) {
            for (id r in data) {
                OcrData *d = (OcrData *) r;
                NSLog(@"%@: %f", d.label, d.accuracy);
                
                NSString *elementText = d.label;
                //识别17位的VIN码
                if (elementText.length == 17 && d.accuracy>0.8) {
                    //正则表达式，排除特殊字符
                    NSString *regex = @"[ABCDEFGHJKLMNPRSTUVWXYZ1234567890]{17}";
                    NSPredicate *test = [NSPredicate predicateWithFormat:@"SELF MATCHES %@", regex];
                    //识别成功
                    if ([test evaluateWithObject:elementText]) {

                        //连续两次识别结果一致，则输出最终结果
                        if ([self->recognizedText isEqualToString:elementText]) {

                            //播放音效
                            NSURL *url=[[NSBundle mainBundle]URLForResource:@"scanSuccess.wav" withExtension:nil];
                            SystemSoundID soundID=8787;
                            AudioServicesCreateSystemSoundID((__bridge CFURLRef)url, &soundID);
                            AudioServicesPlaySystemSound(soundID);
                            
                            //在屏幕上输入结果
                            dispatch_async(dispatch_get_main_queue(), ^{
                                self->textLabel.text = elementText;
                            });
                            
                            NSLog(@"%@",elementText);
                        
                            //停止扫描
                            [self.session stopRunning];
                        
                        }else
                        {
                            //马上再识别一次，对比结果
                            self->recognizedText = elementText;
                            self->isInference = NO;
                        }
                        return;
                    }
                }
            }
        }
        
        //延迟100毫秒再继续识别下一次，降低CPU功耗，省电‼️
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(50 * NSEC_PER_MSEC)), dispatch_get_main_queue(), ^{
            //继续识别
            self->isInference = NO;
        });
    }
}

- (void)viewWillAppear:(BOOL)animated{
    
    [super viewWillAppear:YES];
    
    if (self.session) {
        [self.session startRunning];
    }
}

- (void)viewDidDisappear:(BOOL)animated{
    
    [super viewDidDisappear:YES];
    
    if (self.session) {
        [self.session stopRunning];
    }
    
    [device removeObserver:self forKeyPath:@"adjustingFocus" context:nil];
}

/**
 完成按钮点击事件

 @param sender 按钮
 */
- (void)clickedFinishBtn:(UIButton *)sender {
    
    if (self.delegate && [self.delegate respondsToSelector:@selector(recognitionComplete:)]) {
        [self.delegate recognitionComplete:textLabel.text];
    }
    
    [self.navigationController popViewControllerAnimated:YES];
}

/**
 * 把 CMSampleBufferRef 转化成 UIImage 的方法，参考自：
 * https://stackoverflow.com/questions/19310437/convert-cmsamplebufferref-to-uiimage-with-yuv-color-space
 * note1 : SDK要求 colorSpace 为 CGColorSpaceCreateDeviceRGB
 * note2 : SDK需要 ARGB 格式的图片
 */
- (UIImage *) imageFromSamplePlanerPixelBuffer:(CMSampleBufferRef)sampleBuffer{
    @autoreleasepool {
        // Get a CMSampleBuffer's Core Video image buffer for the media data
        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        // Lock the base address of the pixel buffer
        CVPixelBufferLockBaseAddress(imageBuffer, 0);
        
        // Get the number of bytes per row for the plane pixel buffer
        void *baseAddress = CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0);
        
        // Get the number of bytes per row for the plane pixel buffer
        size_t bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(imageBuffer,0);
        // Get the pixel buffer width and height
        size_t width = CVPixelBufferGetWidth(imageBuffer);
        size_t height = CVPixelBufferGetHeight(imageBuffer);
        
        // Create a device-dependent RGB color space
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        
        // Create a bitmap graphics context with the sample buffer data
        CGContextRef context = CGBitmapContextCreate(baseAddress, width, height, 8,
                                                     bytesPerRow, colorSpace, kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
        // Create a Quartz image from the pixel data in the bitmap graphics context
        CGImageRef quartzImage = CGBitmapContextCreateImage(context);
        // Unlock the pixel buffer
        CVPixelBufferUnlockBaseAddress(imageBuffer,0);
        
        // Free up the context and color space
        CGContextRelease(context);
        CGColorSpaceRelease(colorSpace);
        
        // Create an image object from the Quartz image
        UIImage *image = [UIImage imageWithCGImage:quartzImage];
        
        // Release the Quartz image
        CGImageRelease(quartzImage);
        return (image);
    }
}

/*
#pragma mark - Navigation

// In a storyboard-based application, you will often want to do a little preparation before navigation
- (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender {
    // Get the new view controller using [segue destinationViewController].
    // Pass the selected object to the new view controller.
}
*/

@end
