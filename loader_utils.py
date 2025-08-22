from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import datasets
from models.victim import Victim
from models.Simodel import *


def load_victim_dataset(cfg, dataset_name, model_arch, img_size):
    test_datasets = datasets.__dict__.keys()
    if dataset_name not in test_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(test_datasets))
    dataset = datasets.__dict__[dataset_name]

    # load model family
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    print('modelfamily: ', modelfamily)
    
    # load victim normalization transform as per victim model family
    if dataset_name == 'CIFAR10' and model_arch == 'cnn32':
        print("No normalization in test data")
        victim_mean_std = {'mean': (0.0,), 'std': (1.0,),}
    else:
        victim_mean_std = datasets.modelfamily_to_mean_std[modelfamily]
    victim_normalization_transform = transforms.Compose([transforms.Normalize(mean=victim_mean_std['mean'],
                                 std=victim_mean_std['std'])])
    
    resize_transform = transforms.Resize((img_size, img_size))
    # load test dataset
    test_transform = datasets.modelfamily_to_transforms_sans_normalization[modelfamily]['test'] 
    #   Remove below line for 32 img size 
    test_transform = transforms.Compose([
        test_transform,  # Add the existing transformations
        resize_transform,
    ])
     
    print('dataset', dataset) 
    testset = dataset(cfg, train=False, transform=test_transform) 
    try:
        n_classes = len(testset.classes)  
    except:
        n_classes = testset.n_classes

    return testset, victim_normalization_transform, n_classes


def load_victim_model(arch, model_path, linear_probing, normalization_transform, n_classes):
    # Define architecture
    if arch == 'vgg16':
        from torchvision.models import vgg16
        target_model = vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        target_model.classifier[6] = nn.Linear(in_features=4096, out_features=n_classes)
        linear_keyword = 'classifier.6'
        
    elif arch == 'mobilenet_v2':
        from torchvision.models import mobilenet_v2
        target_model = mobilenet_v2(weights='IMAGENET1K_V1')
        target_model.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes,bias=True)
        linear_keyword = 'classifier.1'
        
    elif arch == 'efficientnet_b0':
        from torchvision.models import efficientnet_b0
        target_model = efficientnet_b0(weights='IMAGENET1K_V1')
        target_model.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes,bias=True)
        linear_keyword = 'classifier.1'
        
    elif arch == 'resnet18':
        import torchvision.models as models
        target_model = models.resnet18(weights="DEFAULT")
        target_model.fc = nn.Linear(in_features=512, out_features=n_classes,bias=True)   
        linear_keyword = 'fc'
        
    elif arch == 'resnet34':
        import torchvision.models as models
        target_model = models.resnet34(weights="DEFAULT")
        target_model.fc = nn.Linear(in_features=512, out_features=n_classes,bias=True)
        linear_keyword = 'fc'
        
    elif arch == 'resnet50':
        import torchvision.models as models
        target_model = models.resnet50(weights="DEFAULT")
        target_model.fc = nn.Linear(in_features=2048, out_features=n_classes,bias=True)
        linear_keyword = 'fc'
        
    elif arch == 'resnet101':
        import torchvision.models as models
        target_model = models.resnet101(weights="DEFAULT")
        target_model.fc = nn.Linear(in_features=2048, out_features=n_classes,bias=True)
        linear_keyword = 'fc'
        
    elif arch == 'resnet152':
        import torchvision.models as models
        target_model = models.resnet152(weights="DEFAULT")
        target_model.fc = nn.Linear(in_features=2048, out_features=n_classes,bias=True)
        linear_keyword = 'fc'
       
    elif arch == 'wide_resnet_50':
        import timm
        target_model = timm.create_model('wide_resnet50_2', pretrained=True,num_classes=n_classes)
        linear_keyword = 'fc'
        
    elif arch == 'densenet121':
        import timm
        target_model = timm.create_model('densenet121', pretrained=True,num_classes=n_classes)
        linear_keyword = 'classifier'
        
    elif arch == 'convnext_base':
        import timm
        target_model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True,num_classes=n_classes)
        linear_keyword = 'head.fc'
        
    elif arch == 'swin_base':
        import timm
        target_model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True,num_classes=n_classes)
        linear_keyword = 'head.fc'
        
    elif arch == 'mvit_base':
        import timm
        target_model = timm.create_model('mvitv2_base_cls.fb_inw21k', pretrained=True,num_classes=n_classes)
        linear_keyword = 'head.fc'
    
    elif arch =='vit_t_16_21k':
        import timm
        target_model = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=False,num_classes=n_classes)
        linear_keyword = 'head'
        
    elif arch =='vit_s_16_21k':
        import timm
        target_model = timm.create_model('vit_small_patch16_224_in21k', pretrained=False,num_classes=n_classes)
        linear_keyword = 'head'
        
    elif arch =='vit_b_16_21k':
        import timm
        target_model = timm.create_model('vit_base_patch16_224_in21k', pretrained=False,num_classes=n_classes)
        linear_keyword = 'head'
       
    elif arch =='vit_l_16_21k':
        import timm
        target_model = timm.create_model('vit_large_patch16_224_in21k', pretrained=False,num_classes=n_classes)
        linear_keyword = 'head'

    elif arch =='vit_b_16_clip':
        import timm
        target_model = timm.create_model('vit_base_patch16_clip_224.laion2b_ft_in1k', pretrained=True,num_classes=n_classes)
        linear_keyword = 'head'
        
    elif arch =='vit_b_16_dino':
        import timm
        target_model = timm.create_model('vit_base_patch16_224.dino', pretrained=True,num_classes=n_classes)
        linear_keyword = 'head'
        
    elif arch == 'vit_b_16_clip_openai':
        import timm
        target_model = timm.create_model('timm/vit_base_patch16_clip_224.openai', pretrained=True,num_classes=n_classes)
        linear_keyword = 'head'

    print(target_model.state_dict().keys())
    if linear_probing == True:
        for name, param in target_model.named_parameters():
            if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                param.requires_grad = False
    print(target_model)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if 'state_dict' in state_dict: 
        state_dict=state_dict["state_dict"]
    try:
        state_dict = {key.replace("last_linear", "fc"): value for key, value in state_dict.items()}
        # print(state_dict.keys())
        target_model.load_state_dict(state_dict, strict=True)
    except: 
        # state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        # print(state_dict.keys())
        target_model.load_state_dict(state_dict, strict=True)

    target_model=target_model.cuda()
    print(f"Loaded target model {model_path}")
    
    # Set victim model to use normalization transform internally
    target_model = Victim(target_model, normalization_transform)
    
    return target_model
    
    
def load_thief_dataset(cfg, dataset_name, data_root, target_model, victim_dataset_name, img_size, id_labels_file=None):
    
    imagenet_mean = (0.4843, 0.4830, 0.4802)
    imagenet_std = (0.1329, 0.1430, 0.1511)

    if dataset_name == 'imagenet32':
        dataset = datasets.__dict__["ImageNet32"]
        if victim_dataset_name == 'MNIST':
            teacher_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            student_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

        else :
            teacher_transform = transforms.Compose([
                transforms.Resize((img_size,img_size)),               
            ])
            student_transform = transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.RandomCrop(img_size, pad_if_needed=True),
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(),                                
                # transforms.RandAugment(),
            ])
            print(student_transform)

        thief_data = dataset(data_root, transform=teacher_transform)         
        thief_data_aug = dataset(data_root, transform=student_transform) 

    elif dataset_name == 'imagenet_full':
        dataset = datasets.__dict__["ImageNet1k"]
        teacher_transform = transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),
            ])
        
        student_transform = transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.RandomCrop(img_size, pad_if_needed=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
                transforms.ToTensor(),
            ])

        thief_data = dataset(cfg, target_model, transform=teacher_transform)
        thief_data_aug = dataset(cfg, target_model, transform=student_transform)
        print('student transform: ', student_transform)
    
    else:
        raise AssertionError('invalid thief dataset')
    
    return thief_data, thief_data_aug
        
    
def create_thief_loaders(thief_data, thief_data_aug, labeled_set, val_set, unlabeled_set, batch_size, target_model):
    
    print("replacing labeled set labels with victim labels")
    print('labeled set ', len(labeled_set))
    train_loader = DataLoader(Subset(thief_data, labeled_set), batch_size=batch_size,
                            pin_memory=False, num_workers=4, shuffle=True)
    target_model.eval()
    with torch.no_grad():
        for d, l0, ind0 in tqdm(train_loader):
            d = d.cuda()
            l = target_model(d).argmax(axis=1, keepdim=False)
            l = l.detach().cpu().tolist()
            for ii, jj in enumerate(ind0):
                thief_data_aug.samples[jj] = (thief_data_aug.samples[jj][0], l[ii])
        
    train_loader = DataLoader(Subset(thief_data_aug, labeled_set), batch_size=batch_size,
                            pin_memory=False, num_workers=4, shuffle=True)
    unlabeled_loader = DataLoader(Subset(thief_data_aug, unlabeled_set), batch_size=batch_size, 
                                        pin_memory=False, num_workers=4, shuffle=True)
    
    print("replacing val labels with victim labels")
    val_loader = DataLoader(Subset(thief_data, val_set), batch_size=batch_size, 
                            pin_memory=False, num_workers=4, shuffle=True)
    target_model.eval()
    with torch.no_grad():
        for d,l,ind0 in tqdm(val_loader):
            d = d.cuda()
            l = target_model(d).argmax(axis=1, keepdim=False)
            l = l.detach().cpu().tolist()
            # print(l)
            for ii, jj in enumerate(ind0):
                thief_data.samples[jj] = (thief_data.samples[jj][0], l[ii])
                
    return train_loader, val_loader, unlabeled_loader
        
    
def load_thief_model(cfg, arch, n_classes, pretrained_path, linear_probing):
    
    if arch == 'vgg16':
        from torchvision.models import vgg16
        thief_model = vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        thief_model.classifier[6] = nn.Linear(in_features=4096, out_features=n_classes)
        linear_keyword = 'classifier.6'
        
    elif arch == 'mobilenet_v2':
        from torchvision.models import mobilenet_v2
        thief_model = mobilenet_v2(weights='IMAGENET1K_V1')
        thief_model.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes,bias=True)
        linear_keyword = 'classifier.1'

    elif arch == 'efficientnet_b0':
        # use this model definition for imagenet1k trained weights
        from torchvision.models import efficientnet_b0
        thief_model = efficientnet_b0(weights='IMAGENET1K_V1')
        thief_model.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes,bias=True)
        linear_keyword = 'classifier.1'
       
    elif arch == 'resnet18':
        import torchvision.models as models
        thief_model = models.resnet18(weights="DEFAULT")       
        print(thief_model.state_dict().keys())
        linear_keyword = 'fc'

    elif arch == 'resnet34':
        import torchvision.models as models
        thief_model = models.resnet34(weights="DEFAULT")
        thief_model.fc = nn.Linear(in_features=512, out_features=n_classes,bias=True)
        linear_keyword = 'fc'

    elif arch == 'resnet50':
        import torchvision.models as models
        thief_model = models.resnet50(weights="DEFAULT")
        thief_model.fc = nn.Linear(in_features=2048, out_features=n_classes,bias=True)
        linear_keyword = 'fc'

    elif arch == 'resnet101':
        import torchvision.models as models
        thief_model = models.resnet101(weights="DEFAULT")
        thief_model.fc = nn.Linear(in_features=2048, out_features=n_classes,bias=True)
        linear_keyword = 'fc'
        
    elif arch == 'resnet152':
        import torchvision.models as models
        thief_model = models.resnet152(weights="DEFAULT")
        thief_model.fc = nn.Linear(in_features=2048, out_features=n_classes,bias=True)
        linear_keyword = 'fc'
        
    elif arch =='vit_t_16_21k':
        import timm
        thief_model = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=True,num_classes=n_classes)
        linear_keyword = 'head'
        
    elif arch =='vit_s_16_21k':
        import timm
        thief_model = timm.create_model('vit_small_patch16_224_in21k', pretrained=True,num_classes=n_classes)
        linear_keyword = 'head'
        
    elif arch =='vit_b_16_21k':
        import timm
        thief_model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True,num_classes=n_classes)
        linear_keyword = 'head'
        
    elif arch =='vit_l_16_21k':
        import timm
        thief_model = timm.create_model('vit_large_patch16_224_in21k', pretrained=True,num_classes=n_classes)
        linear_keyword = 'head'
        
    elif arch =='vit_b_16_clip':
        import timm
        thief_model = timm.create_model('vit_base_patch16_clip_224.laion2b_ft_in1k', pretrained=True, num_classes=n_classes)
        linear_keyword = 'head'
        
    elif arch =='vit_b_16_dino':
        import timm
        thief_model = timm.create_model('vit_base_patch16_224.dino', pretrained=True, num_classes=n_classes)
        linear_keyword = 'head'
       
    elif arch =='vit_b_16_clip_openai':
        import timm
        thief_model = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True, num_classes=n_classes)
        linear_keyword = 'head'
       
    elif arch =='mvit_l':
        import timm
        thief_model = timm.create_model('mvitv2_large_cls.fb_inw21k', pretrained=True, num_classes=n_classes)
        linear_keyword = 'head'
       
    print(thief_model.state_dict().keys())
    print(thief_model)  
        
    if linear_probing == True:
        for name, param in thief_model.named_parameters():
            if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                param.requires_grad = False

    if cfg.ACTIVE.USE_PRETRAINED == True:
        print('Thief Model :',thief_model.state_dict().keys())
        # print("Load pretrained model for initializing the thief, from ", pretrained_path)
        thief_state = thief_model.state_dict()
        # print('\nthief keys: ', thief_state.keys())
        pretrained_state = torch.load(pretrained_path) 
        if 'state_dict' in pretrained_state:
            pretrained_state = pretrained_state['state_dict']
        print('\npretrained keys: ', pretrained_state.keys())
        pretrained_state = { k:v for k,v in pretrained_state.items() if k in thief_state and v.size() == thief_state[k].size() }
        print('\npretrained keys: ', pretrained_state.keys())
        thief_state.update(pretrained_state)
        thief_model.load_state_dict(thief_state, strict=True)

    thief_model = thief_model.cuda()
    return thief_model
    
    
